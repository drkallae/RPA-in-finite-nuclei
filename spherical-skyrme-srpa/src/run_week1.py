from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import HFConfig, MeshConfig, Nucleus, RunConfig
from src.hf.scaffold import run_spherical_hf_scaffold
from src.io_utils import write_json
from src.mesh import RadialMesh
from src.physics.operators import isoscalar_e2_external_field
from src.skyrme import get_skyrme


def main() -> None:
    cfg = RunConfig(
        nucleus=Nucleus(Z=8, N=8),  # Oxygen-16
        mesh=MeshConfig(r_max_fm=20.0, n_points=2000),
        hf=HFConfig(max_iter=50, mixing=0.3),
        skyrme_name="SLy4*",  # treated as SLy4 unless you provide distinct params
        output_dir=Path("out/week1_o16_sly4star_e2"),
        channel="E2",
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    mesh = RadialMesh.uniform(cfg.mesh.r_max_fm, cfg.mesh.n_points)
    skyrme = get_skyrme(cfg.skyrme_name)

    hf_res = run_spherical_hf_scaffold(
        mesh=mesh,
        nucleus_Z=cfg.nucleus.Z,
        nucleus_N=cfg.nucleus.N,
        skyrme=skyrme,
        max_iter=cfg.hf.max_iter,
        mixing=cfg.hf.mixing,
    )

    # Simple sanity: verify normalization
    N_num = mesh.integrate_3d(hf_res.rho_n)
    Z_num = mesh.integrate_3d(hf_res.rho_p)
    e2_sanity = isoscalar_e2_external_field(mesh, hf_res.rho_n, hf_res.rho_p)

    np.save(cfg.output_dir / "r_fm.npy", mesh.r)
    np.save(cfg.output_dir / "rho_n_fm-3.npy", hf_res.rho_n)
    np.save(cfg.output_dir / "rho_p_fm-3.npy", hf_res.rho_p)

    write_json(
        cfg.output_dir / "run_config.json",
        cfg,
    )
    write_json(
        cfg.output_dir / "week1_summary.json",
        {
            "converged": hf_res.converged,
            "n_iter": hf_res.n_iter,
            "N_target": cfg.nucleus.N,
            "Z_target": cfg.nucleus.Z,
            "N_numerical": N_num,
            "Z_numerical": Z_num,
            "isoscalar_E2_sanity_integral": e2_sanity,
            "energies_mev": hf_res.energies_mev,
            "notes": [
                "Week 1: this is a scaffold with reasonable initial densities.",
                "Week 2: implement Skyrme mean fields + radial Schr solver + self-consistent iteration.",
            ],
        },
    )

    print("Wrote outputs to:", cfg.output_dir)
    print(f"N numerical = {N_num:.6f} (target {cfg.nucleus.N})")
    print(f"Z numerical = {Z_num:.6f} (target {cfg.nucleus.Z})")
    print(f"Isoscalar E2 sanity integral = {e2_sanity:.6e}")


if __name__ == "__main__":
    main()