from __future__ import annotations

import numpy as np

from src.constants import HBARC_MEV_FM
from src.mesh import RadialMesh

E2_MEV_FM = 1.43996448  # e^2/(4*pi*eps0) in MeV*fm


def coulomb_direct_potential(mesh: RadialMesh, rho_p: np.ndarray) -> np.ndarray:
    """
    Spherical Coulomb (direct/Hartree) potential for protons:
      V(r) = e^2 [ (1/r) ∫_0^r 4π r'^2 rho_p(r') dr'  +  ∫_r^∞ 4π r' rho_p(r') dr' ]
    with rho_p in fm^-3, result in MeV.
    """
    r = mesh.r
    dr = mesh.dr

    # Charge enclosed Q(r) in units of proton charge (dimensionless)
    integrand_inner = 4.0 * np.pi * rho_p * r**2
    Q = np.cumsum(integrand_inner) * dr  # approximate integral

    V = np.zeros_like(r)
    # first term: Q(r)/r (handle r=0 by limit)
    V[1:] += E2_MEV_FM * Q[1:] / r[1:]
    V[0] = V[1]

    # second term: ∫_r^∞ 4π r' rho(r') dr'
    integrand_outer = 4.0 * np.pi * rho_p * r
    tail = np.cumsum(integrand_outer[::-1]) * dr
    tail = tail[::-1]
    V += E2_MEV_FM * tail

    return V


def coulomb_slater_exchange_potential(rho_p: np.ndarray) -> np.ndarray:
    """
    Local Slater approximation:
      Vx(r) = - e^2 * (3/pi)^(1/3) * rho_p(r)^(1/3)
    """
    # avoid rho<0 numerical noise
    rp = np.clip(rho_p, 0.0, None)
    return -E2_MEV_FM * (3.0 / np.pi) ** (1.0 / 3.0) * rp ** (1.0 / 3.0)