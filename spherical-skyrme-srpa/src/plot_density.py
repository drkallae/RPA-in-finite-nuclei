from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    indir = Path("out/week2b_o16_hf_min_t0t3")
    r = np.load(indir / "r_fm.npy")
    rho_n = np.load(indir / "rho_n_fm-3.npy")
    rho_p = np.load(indir / "rho_p_fm-3.npy")

    plt.figure(figsize=(6, 4))
    plt.plot(r, rho_n, label=r"$\rho_n$")
    plt.plot(r, rho_p, label=r"$\rho_p$", linestyle="--")
    plt.xlim(0, 10)
    plt.ylim(0, max(rho_n.max(), rho_p.max()) * 1.1)
    plt.xlabel("r [fm]")
    plt.ylabel(r"$\rho$ [fm$^{-3}$]")
    plt.title("O-16 spherical HF (development)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(indir / "density.png", dpi=160)
    print("Wrote:", indir / "density.png")


if __name__ == "__main__":
    main()