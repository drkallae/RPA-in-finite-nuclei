from __future__ import annotations

import numpy as np

from src.mesh import RadialMesh


def woods_saxon(mesh: RadialMesh, V0_mev: float, R0_fm: float, a_fm: float) -> np.ndarray:
    r = mesh.r
    return -V0_mev / (1.0 + np.exp((r - R0_fm) / a_fm))