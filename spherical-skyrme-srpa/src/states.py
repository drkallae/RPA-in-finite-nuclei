from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SphericalState:
    """
    Spherical single-particle state quantum numbers.
    We will store radial wavefunction u(r) later (u=r*R).
    """
    n: int          # radial quantum number label in your solver convention
    l: int
    j2: int         # 2*j to keep ints (e.g. j=3/2 -> j2=3)
    tz: int         # +1 for neutron, -1 for proton (convention; choose and keep consistent)

    @property
    def j(self) -> float:
        return self.j2 / 2.0

    @property
    def parity(self) -> int:
        return (-1) ** self.l