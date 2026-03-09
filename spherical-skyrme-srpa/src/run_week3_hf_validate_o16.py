from __future__ import annotations

from pathlib import Path

import numpy as np

from src.mesh import RadialMesh
from src.skyrme import get_skyrme
from src.hf.hf_minimal import run_hf_minimal_t0t3


def _half_density_radius(r: np.ndarray, rho: np.ndarray) -> float:
    i0 = int(np.argmax(rho))
    rho_half = 0.5 * float(rho[i0])
    j = int(np.argmin(np.abs(rho - rho_half)))
    return float(r[j])


def _rms_radius_from_density(r: np.ndarray, rho: np.ndarray) -> float:
    # <r^2> = ∫ r^2 rho d^3r / ∫ rho d^3r
    num = 4.0 * np.pi * np.trapezoid(rho * r**4, r)
    den = 4.0 * np.pi * np.trapezoid(rho * r**2, r)
    return float(np.sqrt(num / den))


def main() -> None:
    outdir = Path("out/week3_o16_hf_coulomb_validate")
    outdir.mkdir(parents=True, exist_ok=True)

    mesh = RadialMesh.uniform(r_max_fm=20.0, n_points=300)
    sk = get_skyrme("SLy4")

    C_SURF = 300.0

    res = run_hf_minimal_t0t3(
        mesh=mesh,
        nucleus_Z=8,
        nucleus_N=8,
        skyrme=sk,
        l_max=6,
        n_states_per_l=10,
        max_iter=400,
        mixing=0.10,
        tol_rho=2e-5,
        C_surf=C_SURF,
        use_coulomb=True,
        coulomb_exchange=True,
        use_spin_orbit=True,
        W0=120.0,
    )

    print("Converged:", res.converged, "in iters:", res.n_iter)
    print("N check:", mesh.integrate_3d(res.rho_n))
    print("Z check:", mesh.integrate_3d(res.rho_p))

    rho_tot = res.rho_n + res.rho_p
    print("\nDensity quick checks:")
    print("rho_tot(0) =", float(rho_tot[0]), "fm^-3")
    print("rho_tot,max =", float(rho_tot.max()), "fm^-3")
    print("R_half (total) ~", _half_density_radius(mesh.r, rho_tot), "fm")
    print("rms radii [fm]: r_tot =", _rms_radius_from_density(mesh.r, rho_tot),
          " r_p =", _rms_radius_from_density(mesh.r, res.rho_p))

    def _fmt_orb(o) -> str:
        if getattr(o, "j2", None) is None:
            return f"  l={o.l}  idx={o.n_index}  occ={o.occ:4.1f}  E={o.energy_mev:9.4f} MeV"
        return f"  l={o.l}  j={o.j2}/2  idx={o.n_index}  occ={o.occ:4.1f}  E={o.energy_mev:9.4f} MeV"

    print("\nNeutron orbitals:")
    for o in res.orbitals_n:
            print(_fmt_orb(o))

    print("\nProton orbitals:")
    for o in res.orbitals_p:
            print(_fmt_orb(o))

    np.save(outdir / "r_fm.npy", mesh.r)
    np.save(outdir / "rho_n_fm-3.npy", res.rho_n)
    np.save(outdir / "rho_p_fm-3.npy", res.rho_p)
    np.save(outdir / "U_n_MeV.npy", res.U_n)
    np.save(outdir / "U_p_MeV.npy", res.U_p)
    if res.Uls_n is not None:
        np.save(outdir / "Uls_n_MeV.npy", res.Uls_n)
    if res.Uls_p is not None:
        np.save(outdir / "Uls_p_MeV.npy", res.Uls_p)


if __name__ == "__main__":
    main()