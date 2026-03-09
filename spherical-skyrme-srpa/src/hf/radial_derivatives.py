from __future__ import annotations

import numpy as np

from src.mesh import RadialMesh


def d_dr(mesh: RadialMesh, f: np.ndarray) -> np.ndarray:
    """Central difference derivative df/dr on uniform mesh."""
    dr = mesh.dr
    df = np.zeros_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (2.0 * dr)
    df[0] = 0.0
    df[-1] = (f[-1] - f[-2]) / dr
    return df