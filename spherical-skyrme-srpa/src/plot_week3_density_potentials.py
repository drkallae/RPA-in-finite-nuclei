from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    indir = Path("out/week3_o16_hf_coulomb_validate")

    r = np.load(indir / "r_fm.npy")
    rn = np.load(indir / "rho_n_fm-3.npy")
    rp = np.load(indir / "rho_p_fm-3.npy")
    Un = np.load(indir / "U_n_MeV.npy")
    Up = np.load(indir / "U_p_MeV.npy")

    # ---- Density ----
    plt.figure()
    plt.plot(r, rn + rp, label="rho total", lw=2)
    plt.plot(r, rn, "--", label="rho_n")
    plt.plot(r, rp, ":", label="rho_p")
    plt.xlim(0, 10)
    plt.xlabel("r [fm]")
    plt.ylabel("rho [fm^-3]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(indir / "density.png", dpi=180)

    # ---- Central mean fields ----
    plt.figure()
    plt.plot(r, Un, label="U_n", lw=2)
    plt.plot(r, Up, "--", label="U_p", lw=2)
    plt.axhline(0, color="k", lw=1, alpha=0.4)
    plt.xlim(0, 10)
    plt.xlabel("r [fm]")
    plt.ylabel("U [MeV]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(indir / "potentials.png", dpi=180)

    # ---- Spin-orbit strength (optional) ----
    uls_n_path = indir / "Uls_n_MeV.npy"
    uls_p_path = indir / "Uls_p_MeV.npy"
    if uls_n_path.exists() and uls_p_path.exists():
        Uls_n = np.load(uls_n_path)
        Uls_p = np.load(uls_p_path)

        plt.figure()
        plt.plot(r, Uls_n, label="Uls_n", lw=2)
        plt.plot(r, Uls_p, "--", label="Uls_p", lw=2)
        plt.axhline(0, color="k", lw=1, alpha=0.4)
        plt.xlim(0, 10)
        plt.xlabel("r [fm]")
        plt.ylabel("U_ls strength [MeV]")  # convention-dependent
        plt.legend()
        plt.tight_layout()
        plt.savefig(indir / "spin_orbit_strength.png", dpi=180)
        print("Wrote spin_orbit_strength.png to", indir)

    print("Wrote density.png and potentials.png to", indir)


if __name__ == "__main__":
    main()