from __future__ import annotations

from pathlib import Path

import numpy as np

from src.mesh import RadialMesh
from src.skyrme import get_skyrme
from src.hf.hf_minimal import run_hf_minimal_t0t3
from src.hf.skyrme_meanfield_min import central_mean_field_t0_t3
from src.hf.radial_ops import laplacian_spherical


def main() -> None:
    outdir = Path("out/week2b_o16_hf_min_t0t3")
    outdir.mkdir(parents=True, exist_ok=True)

    mesh = RadialMesh.uniform(r_max_fm=20.0, n_points=200)
    sk = get_skyrme("SLy4")

    res = run_hf_minimal_t0t3(
    mesh=mesh,
    nucleus_Z=8,
    nucleus_N=8,
    skyrme=sk,
    l_max=6,
    n_states_per_l=10,
    max_iter=200,
    mixing=0.10,
    tol_rho=2e-5,
    C_surf=60.0,
    use_coulomb=False,
)

    print("Converged:", res.converged, "in iters:", res.n_iter)
    print("N check:", mesh.integrate_3d(res.rho_n))
    print("Z check:", mesh.integrate_3d(res.rho_p))

    print("\nNeutron orbitals (no spin-orbit):")
    for o in res.orbitals_n:
        print(f"  l={o.l}  idx={o.n_index}  occ={o.occ:4.1f}  E={o.energy_mev:9.4f} MeV")

    print("\nProton orbitals (no spin-orbit):")
    for o in res.orbitals_p:
        print(f"  l={o.l}  idx={o.n_index}  occ={o.occ:4.1f}  E={o.energy_mev:9.4f} MeV")

    np.save(outdir / "r_fm.npy", mesh.r)
    np.save(outdir / "rho_n_fm-3.npy", res.rho_n)
    np.save(outdir / "rho_p_fm-3.npy", res.rho_p)



    # Rebuild and save mean fields used in the final iteration (including surface term)
    U_n = central_mean_field_t0_t3(sk, rho_n=res.rho_n, rho_p=res.rho_p, tz=+1)
    U_p = central_mean_field_t0_t3(sk, rho_n=res.rho_n, rho_p=res.rho_p, tz=-1)
    U_n -= U_n[-1]
    U_p -= U_p[-1]

    C_surf = 60.0
    rho = res.rho_n + res.rho_p
    U_surf = -C_surf * laplacian_spherical(mesh, rho)
    U_n = U_n + U_surf
    U_p = U_p + U_surf
    U_n -= U_n[-1]
    U_p -= U_p[-1]

    np.save(outdir / "U_n_MeV.npy", U_n)
    np.save(outdir / "U_p_MeV.npy", U_p)
   

if __name__ == "__main__":
    main()