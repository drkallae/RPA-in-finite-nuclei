from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.mesh import RadialMesh
from src.skyrme import SkyrmeParams
from src.hf.radial_solve import solve_radial_bound_states_box
from src.hf.skyrme_meanfield_min import central_mean_field_t0_t3
from src.hf.skyrme_meanfield_t1t2 import B_field_t1t2
from src.hf.radial_ops import laplacian_spherical
from src.hf.coulomb import coulomb_direct_potential, coulomb_slater_exchange_potential
from src.hf.radial_derivatives import d_dr
from src.hf.fill import (
    Orbital,
    density_from_orbitals,
    fill_spherical_no_spin_orbit,
    fill_spherical_jj,
)
from src.hf.densities import tau_from_orbitals


@dataclass
class HFMinimalResult:
    converged: bool
    n_iter: int
    rho_n: np.ndarray
    rho_p: np.ndarray
    orbitals_n: list[Orbital]
    orbitals_p: list[Orbital]
    U_n: np.ndarray
    U_p: np.ndarray
    Uls_n: np.ndarray | None = None
    Uls_p: np.ndarray | None = None
    tau_n: np.ndarray | None = None
    tau_p: np.ndarray | None = None


def _ls_expectation(l: int, j2: int) -> float:
    """Return <l·s> for given l and j2=2j."""
    j = 0.5 * j2
    return 0.5 * (j * (j + 1.0) - l * (l + 1.0) - 0.75)


def _couplings_C_laprho(sk: SkyrmeParams) -> tuple[float, float]:
    """
    Return (C0_laprho, C1_laprho) for a Laplacian-form gradient sector:
        H ⊃ C0_laprho * rho * Δrho + C1_laprho * rho1 * Δrho1

    NOTE: Conventions can differ by integration by parts vs (∇rho)^2.
    If results look unphysical, this mapping is the first thing to revisit.
    """
    t1, t2, x1, x2 = sk.t1, sk.t2, sk.x1, sk.x2
    C0_lap = (1.0 / 64.0) * (-9.0 * t1 + t2 * (5.0 + 4.0 * x2))
    C1_lap = (1.0 / 64.0) * (3.0 * t1 * (1.0 + 2.0 * x1) + t2 * (1.0 + 2.0 * x2))
    return C0_lap, C1_lap


def run_hf_minimal_t0t3(
    mesh: RadialMesh,
    nucleus_Z: int,
    nucleus_N: int,
    skyrme: SkyrmeParams,
    l_max: int = 6,
    n_states_per_l: int = 8,
    max_iter: int = 80,
    mixing: float = 0.3,
    tol_rho: float = 1e-5,
    *,
    # With an explicit t1/t2 gradient sector, C_surf should be 0 (or very small).
    C_surf: float = 0.0,
    # Scale the Skyrme gradient (Laplacian) contribution separately:
    k_lap0: float = 1.2,  # isoscalar piece (Δrho)
    k_lap1: float = 1.0,  # isovector piece (Δrho1) -- tune to fix Ca-40 vs Ca-48 isotope shift
    use_coulomb: bool = False,
    coulomb_exchange: bool = True,
    use_spin_orbit: bool = False,
    W0: float | None = None,
) -> HFMinimalResult:
    if W0 is None:
        W0 = skyrme.W0

    r = mesh.r

    # Initial densities (Fermi-ish) scaled to N and Z
    rho0 = 0.16
    r0 = 1.2 * (nucleus_Z + nucleus_N) ** (1.0 / 3.0)
    a = 0.5
    shape = 1.0 / (1.0 + np.exp((r - r0) / a))

    rho_n = rho0 * shape
    rho_p = rho0 * shape
    rho_n *= nucleus_N / mesh.integrate_3d(rho_n)
    rho_p *= nucleus_Z / mesh.integrate_3d(rho_p)

    # τ state carried between iterations (currently NOT used in U; kept for future work)
    tau_n = np.zeros_like(r)
    tau_p = np.zeros_like(r)

    # Placeholders to ensure defined at return
    orbs_n: list[Orbital] = []
    orbs_p: list[Orbital] = []
    U_n = np.zeros_like(r)
    U_p = np.zeros_like(r)
    Uls_n: np.ndarray | None = None
    Uls_p: np.ndarray | None = None

    # Precompute Skyrme couplings used in the added t1/t2 gradient pieces
    C0_lap, C1_lap = _couplings_C_laprho(skyrme)

    for it in range(1, max_iter + 1):
        # --- build mean fields from current densities ---
        U_n = central_mean_field_t0_t3(skyrme, rho_n=rho_n, rho_p=rho_p, tz=+1)
        U_p = central_mean_field_t0_t3(skyrme, rho_n=rho_n, rho_p=rho_p, tz=-1)

        # t1/t2 effective mass field B(r) (includes ħ²/2m baseline in your B_field_t1t2)
        B_n = B_field_t1t2(skyrme, rho_n=rho_n, rho_p=rho_p, tz=+1)
        B_p = B_field_t1t2(skyrme, rho_n=rho_n, rho_p=rho_p, tz=-1)
        B_n = np.maximum(B_n, 1e-6)
        B_p = np.maximum(B_p, 1e-6)

        # Gradient (Laplacian) sector
        rho = rho_n + rho_p
        rho1 = rho_n - rho_p

        lap_rho_n = laplacian_spherical(mesh, rho_n)
        lap_rho_p = laplacian_spherical(mesh, rho_p)
        lap_rho = lap_rho_n + lap_rho_p
        lap_rho1 = lap_rho_n - lap_rho_p

        # U_lap: δ/δrho_q of (C0_lap rho Δrho + C1_lap rho1 Δrho1)
        #        -> U_q ⊃ 2 C0_lap Δrho + 2 (±) C1_lap Δrho1
        U_lap_n = 2.0 * k_lap0 * C0_lap * lap_rho + 2.0 * k_lap1 * (+1.0) * C1_lap * lap_rho1
        U_lap_p = 2.0 * k_lap0 * C0_lap * lap_rho + 2.0 * k_lap1 * (-1.0) * C1_lap * lap_rho1

        U_n = U_n + U_lap_n
        U_p = U_p + U_lap_p

        # Optional pragmatic surface stabilization (leave off unless needed)
        if C_surf != 0.0:
            lap_rho_tot = laplacian_spherical(mesh, rho)
            U_surf = -C_surf * lap_rho_tot
            U_n = U_n + U_surf
            U_p = U_p + U_surf

        # Gauge-fix: shift so U(Rmax)=0 before Coulomb/SO
        U_n = U_n - U_n[-1]
        U_p = U_p - U_p[-1]

        # Coulomb (protons only)
        if use_coulomb:
            U_p = U_p + coulomb_direct_potential(mesh, rho_p)
            if coulomb_exchange:
                U_p = U_p + coulomb_slater_exchange_potential(rho_p)

        # Keep box edge at 0 after Coulomb
        U_n = U_n - U_n[-1]
        U_p = U_p - U_p[-1]

        # Spin-orbit radial strengths (optional)
        Uls_n_arr = np.zeros_like(r)
        Uls_p_arr = np.zeros_like(r)
        if use_spin_orbit:
            drho = d_dr(mesh, rho)
            drho_n = d_dr(mesh, rho_n)
            drho_p = d_dr(mesh, rho_p)

            invr = np.zeros_like(r)
            invr[1:] = 1.0 / r[1:]
            invr[0] = invr[1]

            Uls_n_arr = 0.5 * W0 * invr * (drho + drho_n)
            Uls_p_arr = 0.5 * W0 * invr * (drho + drho_p)

        Uls_n = Uls_n_arr if use_spin_orbit else None
        Uls_p = Uls_p_arr if use_spin_orbit else None

        if it == 1 or it % 5 == 0:
            print(
                f"[it={it:03d}] Un,min={U_n.min():+.3f} Up,min={U_p.min():+.3f}  "
                f"rho_max={float(rho.max()):.4f}  k_lap0={k_lap0:.2f} k_lap1={k_lap1:.2f}"
            )

        # --- solve orbitals ---
        if not use_spin_orbit:
            eig_n: dict[int, list[tuple[float, np.ndarray]]] = {}
            eig_p: dict[int, list[tuple[float, np.ndarray]]] = {}
            for l in range(0, l_max + 1):
                eig_n[l] = solve_radial_bound_states_box(
                    mesh, U_mev=U_n, l=l, tz=+1, n_states=n_states_per_l, B_mev_fm2=B_n
                )
                eig_p[l] = solve_radial_bound_states_box(
                    mesh, U_mev=U_p, l=l, tz=-1, n_states=n_states_per_l, B_mev_fm2=B_p
                )

            orbs_n = fill_spherical_no_spin_orbit(tz=+1, Nq=nucleus_N, eigen_by_l=eig_n)
            orbs_p = fill_spherical_no_spin_orbit(tz=-1, Nq=nucleus_Z, eigen_by_l=eig_p)
        else:
            eig_n_lj: dict[tuple[int, int], list[tuple[float, np.ndarray]]] = {}
            eig_p_lj: dict[tuple[int, int], list[tuple[float, np.ndarray]]] = {}

            for l in range(0, l_max + 1):
                j2_list = [1] if l == 0 else [2 * l - 1, 2 * l + 1]
                for j2 in j2_list:
                    ls_fac = _ls_expectation(l, j2)
                    Ueff_n = U_n + ls_fac * Uls_n_arr
                    Ueff_p = U_p + ls_fac * Uls_p_arr

                    eig_n_lj[(l, j2)] = solve_radial_bound_states_box(
                        mesh,
                        U_mev=Ueff_n,
                        l=l,
                        tz=+1,
                        n_states=n_states_per_l,
                        B_mev_fm2=B_n,
                    )
                    eig_p_lj[(l, j2)] = solve_radial_bound_states_box(
                        mesh,
                        U_mev=Ueff_p,
                        l=l,
                        tz=-1,
                        n_states=n_states_per_l,
                        B_mev_fm2=B_p,
                    )

            orbs_n = fill_spherical_jj(tz=+1, Nq=nucleus_N, eigen_by_lj=eig_n_lj)
            orbs_p = fill_spherical_jj(tz=-1, Nq=nucleus_Z, eigen_by_lj=eig_p_lj)

        # --- build new densities and tau from orbitals ---
        rho_n_new = density_from_orbitals(mesh, orbs_n)
        rho_p_new = density_from_orbitals(mesh, orbs_p)
        tau_n_new = tau_from_orbitals(mesh, orbs_n)
        tau_p_new = tau_from_orbitals(mesh, orbs_p)

        # Mix densities and tau (tau not used in U yet, but harmless)
        rho_n_m = (1.0 - mixing) * rho_n + mixing * rho_n_new
        rho_p_m = (1.0 - mixing) * rho_p + mixing * rho_p_new
        tau_n_m = (1.0 - mixing) * tau_n + mixing * tau_n_new
        tau_p_m = (1.0 - mixing) * tau_p + mixing * tau_p_new

        # Enforce particle numbers
        rho_n_m *= nucleus_N / mesh.integrate_3d(rho_n_m)
        rho_p_m *= nucleus_Z / mesh.integrate_3d(rho_p_m)

        # Convergence check on densities
        dn = float(np.max(np.abs(rho_n_m - rho_n)))
        dp = float(np.max(np.abs(rho_p_m - rho_p)))
        drho = max(dn, dp)

        if it == 1 or it % 5 == 0:
            print(f"[it={it:03d}] drho={drho:.3e}")

        rho_n, rho_p = rho_n_m, rho_p_m
        tau_n, tau_p = tau_n_m, tau_p_m

        if drho < tol_rho:
            return HFMinimalResult(
                converged=True,
                n_iter=it,
                rho_n=rho_n,
                rho_p=rho_p,
                orbitals_n=orbs_n,
                orbitals_p=orbs_p,
                U_n=U_n,
                U_p=U_p,
                Uls_n=Uls_n,
                Uls_p=Uls_p,
                tau_n=tau_n,
                tau_p=tau_p,
            )

    return HFMinimalResult(
        converged=False,
        n_iter=max_iter,
        rho_n=rho_n,
        rho_p=rho_p,
        orbitals_n=orbs_n,
        orbitals_p=orbs_p,
        U_n=U_n,
        U_p=U_p,
        Uls_n=Uls_n,
        Uls_p=Uls_p,
        tau_n=tau_n,
        tau_p=tau_p,
    )