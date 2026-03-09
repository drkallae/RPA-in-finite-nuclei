from __future__ import annotations

import numpy as np

from src.mesh import RadialMesh
from src.hf.fill import Orbital
from src.hf.radial_derivatives import d_dr


def tau_from_orbitals(mesh: RadialMesh, orbitals: list[Orbital]) -> np.ndarray:
    """
    Time-even kinetic density tau(r) in spherical symmetry.

    Conventions in this project:
      u(r) normalized: ∫ |u|^2 dr = 1
      R(r) = u(r)/r
      rho(r) = Σ occ * |R(r)|^2 / (4π)

    Then:
      tau(r) = Σ occ/(4π) * ( |dR/dr|^2 + l(l+1) |R|^2 / r^2 )

    Regularization: at r=0 copy from r[1].
    """
    r = mesh.r
    tau = np.zeros_like(r)

    for orb in orbitals:
        u = orb.u
        l = orb.l

        R = np.zeros_like(u)
        R[1:] = u[1:] / r[1:]
        R[0] = R[1]

        dR = d_dr(mesh, R)

        term = dR * dR
        term[1:] += (l * (l + 1.0)) * (R[1:] * R[1:]) / (r[1:] * r[1:])
        term[0] = term[1]

        tau += (orb.occ / (4.0 * np.pi)) * term

    tau[0] = tau[1]
    return tau