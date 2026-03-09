from __future__ import annotations

import numpy as np

from src.mesh import RadialMesh


def laplacian_spherical(mesh: RadialMesh, f: np.ndarray) -> np.ndarray:
    """
    Spherical Laplacian:
      ∇²f = (1/r²) d/dr ( r² df/dr )

    Finite-difference on a uniform mesh.
    Boundary handling:
      - at r=0: enforce symmetry df/dr=0
      - at r=Rmax: one-sided derivative, adequate for box.
    """
    r = mesh.r
    dr = mesh.dr
    n = r.size
    out = np.zeros_like(f)

    # df/dr (central)
    df = np.zeros_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (2.0 * dr)
    df[0] = 0.0  # symmetry at origin
    df[-1] = (f[-1] - f[-2]) / dr

    g = r * r * df

    dg = np.zeros_like(f)
    dg[1:-1] = (g[2:] - g[:-2]) / (2.0 * dr)
    dg[0] = 0.0
    dg[-1] = (g[-1] - g[-2]) / dr

    out[1:] = dg[1:] / (r[1:] * r[1:])
    out[0] = out[1]
    return out