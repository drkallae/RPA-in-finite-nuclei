from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

from src.constants import HBAR2_OVER_2MN, HBAR2_OVER_2MP
from src.mesh import RadialMesh


@dataclass(frozen=True)
class BoundState:
    energy_mev: float
    u: np.ndarray  # radial wavefunction u(r) = r * R(r), normalized to ∫|u|^2 dr = 1
    l: int
    j2: int
    tz: int  # +1 neutron, -1 proton


def _hbar2_over_2m(tz: int) -> float:
    return HBAR2_OVER_2MN if tz == +1 else HBAR2_OVER_2MP


def solve_radial_bound_states_box(
    mesh: RadialMesh,
    U_mev: np.ndarray,
    l: int,
    tz: int,
    n_states: int = 6,
    *,
    B_mev_fm2: np.ndarray | None = None,
) -> list[tuple[float, np.ndarray]]:
    """
    Solve a radial bound-state problem in a spherical box on a uniform mesh.

    Equation for u(r) = r R(r):
        [ - d/dr ( B(r) d/dr ) + B(r) l(l+1)/r^2 + U(r) ] u(r) = E u(r)

    where:
      - U(r) is in MeV
      - B(r) is in MeV*fm^2
      - for constant mass: B(r) = ħ²/(2m)

    Boundary conditions (box):
      u(0) = 0, u(Rmax) = 0

    Returns the lowest n_states eigenpairs (E, u) with u normalized to ∫|u|^2 dr = 1.
    """
    r = mesh.r
    dr = mesh.dr
    n = r.size

    if U_mev.shape != r.shape:
        raise ValueError("U_mev shape mismatch")

    # Interior points 1..n-2 enforce u(0)=u(Rmax)=0
    ri = r[1:-1]
    Ui = U_mev[1:-1]
    ni = ri.size

    # Choose B(r) (MeV fm^2)
    if B_mev_fm2 is None:
        B = _hbar2_over_2m(tz) * np.ones_like(r)
    else:
        if B_mev_fm2.shape != r.shape:
            raise ValueError("B_mev_fm2 shape mismatch")
        B = np.maximum(B_mev_fm2, 1e-6)

    # Midpoints: B_half[i] = B_{i+1/2} for i=0..n-2
    B_half = 0.5 * (B[:-1] + B[1:])  # shape (n-1,)

    # Interior global indices i = 1..n-2 (count ni = n-2)
    # For each interior i:
    #   B_{i-1/2} = B_half[i-1]
    #   B_{i+1/2} = B_half[i]
    B_minus = B_half[0:ni]       # length ni
    B_plus = B_half[1 : ni + 1]  # length ni

    # Tridiagonal kinetic operator for -d/dr (B du/dr)
    diag = (B_minus + B_plus) / (dr * dr)   # length ni
    off = (-B_plus[:-1]) / (dr * dr)        # length ni-1

    # Add diagonal potential + centrifugal term using Bi = B(r_i)
    Bi = B[1:-1]
    Vcent = Bi * (l * (l + 1.0)) / (ri * ri)
    diag = diag + (Ui + Vcent)

    # Solve only the lowest n_states eigenpairs (fast)
    m = min(n_states, ni)
    evals, evecs = la.eigh_tridiagonal(
        diag,
        off,
        select="i",
        select_range=(0, m - 1),
    )

    out: list[tuple[float, np.ndarray]] = []
    for k in range(min(n_states, evals.size)):
        E = float(evals[k])
        u_i = evecs[:, k].copy()

        # Normalize: ∫ |u|^2 dr = 1 (interior integral is enough since boundaries are zero)
        norm = np.sqrt(np.trapezoid(u_i * u_i, ri))
        u_i /= norm

        # Expand to full array with boundary zeros
        u = np.zeros(n)
        u[1:-1] = u_i
        out.append((E, u))

    return out