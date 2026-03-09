from __future__ import annotations

from pathlib import Path

import numpy as np

from src.mesh import RadialMesh
from src.skyrme import get_skyrme
from src.hf.hf_minimal import run_hf_minimal_t0t3


def half_density_radius(r: np.ndarray, rho: np.ndarray) -> float:
    i0 = int(np.argmax(rho))
    rho_half = 0.5 * float(rho[i0])
    j = int(np.argmin(np.abs(rho - rho_half)))
    return float(r[j])


def rms_radius_from_density(r: np.ndarray, rho: np.ndarray) -> float:
    num = 4.0 * np.pi * np.trapezoid(rho * r**4, r)
    den = 4.0 * np.pi * np.trapezoid(rho * r**2, r)
    return float(np.sqrt(num / den))


def print_occupied(orbs, label: str) -> None:
    print(f"\n{label} orbitals (occupied):")
    for o in orbs:
        if o.occ <= 0:
            continue
        if getattr(o, "j2", None) is None:
            print(f"  l={o.l} idx={o.n_index} occ={o.occ:4.1f}  E={o.energy_mev:9.4f} MeV")
        else:
            print(f"  l={o.l} j={o.j2}/2 idx={o.n_index} occ={o.occ:4.1f}  E={o.energy_mev:9.4f} MeV")


def main() -> None:
    Z, N = 20, 28
    tag = "ca48"

    outdir = Path("out") / "hf_ca48_run"
    outdir.mkdir(parents=True, exist_ok=True)

    # Mesh: adjust if you want (bigger box for safety)
    mesh = RadialMesh.uniform(r_max_fm=20.0, n_points=300)

    sk = get_skyrme("SLy4")

    res = run_hf_minimal_t0t3(
        mesh=mesh,
        nucleus_Z=Z,
        nucleus_N=N,
        skyrme=sk,
        l_max=6,
        n_states_per_l=10,
        max_iter=400,
        mixing=0.05,
        tol_rho=2e-5,
        C_surf=0.0,               # with t1/t2 gradient sector in hf_minimal, keep this 0
        use_coulomb=True,
        coulomb_exchange=True,
        use_spin_orbit=True,
        W0=sk.W0,                 # keep consistent with parameter set
    )

    print("\n" + "=" * 72)
    print(f"{tag.upper()}  Z={Z}  N={N}")
    print("=" * 72)
    print("Converged:", res.converged, "in iters:", res.n_iter)
    print("N check:", mesh.integrate_3d(res.rho_n))
    print("Z check:", mesh.integrate_3d(res.rho_p))

    rho_tot = res.rho_n + res.rho_p
    print("\nDensity quick checks:")
    print("rho_tot(0) =", float(rho_tot[0]), "fm^-3")
    print("rho_tot,max =", float(rho_tot.max()), "fm^-3")
    print("R_half (total) ~", half_density_radius(mesh.r, rho_tot), "fm")
    print(
        "rms radii [fm]: r_tot =",
        rms_radius_from_density(mesh.r, rho_tot),
        " r_p =",
        rms_radius_from_density(mesh.r, res.rho_p),
    )

    print_occupied(res.orbitals_n, "Neutron")
    print_occupied(res.orbitals_p, "Proton")

    # Save arrays
    np.save(outdir / "r_fm.npy", mesh.r)
    np.save(outdir / "rho_n_fm-3.npy", res.rho_n)
    np.save(outdir / "rho_p_fm-3.npy", res.rho_p)
    np.save(outdir / "rho_tot_fm-3.npy", rho_tot)

    np.save(outdir / "U_n_MeV.npy", res.U_n)
    np.save(outdir / "U_p_MeV.npy", res.U_p)

    if res.Uls_n is not None:
        np.save(outdir / "Uls_n_MeV.npy", res.Uls_n)
    if res.Uls_p is not None:
        np.save(outdir / "Uls_p_MeV.npy", res.Uls_p)

    if getattr(res, "tau_n", None) is not None and res.tau_n is not None:
        np.save(outdir / "tau_n_fm-5.npy", res.tau_n)
    if getattr(res, "tau_p", None) is not None and res.tau_p is not None:
        np.save(outdir / "tau_p_fm-5.npy", res.tau_p)

    print(f"\nSaved outputs to: {outdir}")


if __name__ == "__main__":
    main()