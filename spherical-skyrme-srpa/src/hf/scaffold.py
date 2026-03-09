from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.mesh import RadialMesh
from src.skyrme import SkyrmeParams


@dataclass
class HFResult:
    converged: bool
    n_iter: int
    # Placeholders:
    rho_n: np.ndarray
    rho_p: np.ndarray
    energies_mev: dict


def run_spherical_hf_scaffold(
    mesh: RadialMesh,
    nucleus_Z: int,
    nucleus_N: int,
    skyrme: SkyrmeParams,
    max_iter: int,
    mixing: float,
) -> HFResult:
    """
    Week 1 scaffold:
    - Creates reasonable initial densities (Fermi shapes)
    - Iteration loop exists, but mean fields + Schr eq not yet implemented
    """
    r = mesh.r

    # crude initial densities (fm^-3): normalized approximately by scaling later
    def fermi(r0: float, a: float) -> np.ndarray:
        return 1.0 / (1.0 + np.exp((r - r0) / a))

    rho0 = 0.16  # saturation-ish
    r0 = 1.2 * (nucleus_Z + nucleus_N) ** (1.0 / 3.0)
    a = 0.5
    shape = fermi(r0=r0, a=a)

    rho_n = rho0 * shape
    rho_p = rho0 * shape

    # scale to correct particle numbers (approx): N = ∫ rho_n d^3r
    n_now = mesh.integrate_3d(rho_n)
    p_now = mesh.integrate_3d(rho_p)
    rho_n *= nucleus_N / n_now
    rho_p *= nucleus_Z / p_now

    # "Iteration" placeholder: just keep densities fixed for now.
    for it in range(1, max_iter + 1):
        # later: build mean fields from rho_n, rho_p, solve for orbitals, recompute densities
        # mixing would be applied between old/new densities
        pass

    energies = {
        "E_total_MeV": float("nan"),
        "E_kin_MeV": float("nan"),
        "E_skyrme_MeV": float("nan"),
        "E_coul_MeV": float("nan"),
        "notes": "Week 1 scaffold: energies not implemented.",
        "skyrme_t0": skyrme.t0,
        "skyrme_t3": skyrme.t3,
        "mixing": mixing,
    }

    return HFResult(converged=False, n_iter=max_iter, rho_n=rho_n, rho_p=rho_p, energies_mev=energies)