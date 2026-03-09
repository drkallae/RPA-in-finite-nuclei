from __future__ import annotations

import numpy as np

from src.mesh import RadialMesh


def e2_radial_form_factor(mesh: RadialMesh) -> np.ndarray:
    """
    Radial dependence for the E2 operator ~ r^2 Y_2m.
    In spherical reduced matrix elements you'll factor angular parts separately.
    """
    return mesh.r**2


def isoscalar_e2_external_field(mesh: RadialMesh, rho_n: np.ndarray, rho_p: np.ndarray) -> float:
    """
    Placeholder 'sanity' quantity: ∫ (rho_n+rho_p) r^2 d^3r.
    Not the actual B(E2) etc—just something to track.
    """
    f = (rho_n + rho_p) * mesh.r**2
    return mesh.integrate_3d(f)