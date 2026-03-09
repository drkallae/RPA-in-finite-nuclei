from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadialMesh:
    r: np.ndarray  # fm
    dr: float

    @staticmethod
    def uniform(r_max_fm: float, n_points: int) -> "RadialMesh":
        if n_points < 10:
            raise ValueError("n_points too small")
        r = np.linspace(0.0, r_max_fm, n_points)
        dr = float(r[1] - r[0])
        return RadialMesh(r=r, dr=dr)

    def integrate_3d(self, f: np.ndarray) -> float:
        """
        Integrate scalar radial function f(r) over 3D:
          ∫ f(r) d^3r = 4π ∫_0^∞ f(r) r^2 dr
        """
        if f.shape != self.r.shape:
            raise ValueError("shape mismatch")
        return float(4.0 * np.pi * np.trapezoid(f * self.r**2, self.r))