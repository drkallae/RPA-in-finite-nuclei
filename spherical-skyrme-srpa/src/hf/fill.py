from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.mesh import RadialMesh


@dataclass(frozen=True)
class Orbital:
    tz: int         # +1 neutron, -1 proton
    l: int
    j2: int | None  # 2*j (e.g. 1,3,5,...). None means "no spin-orbit / j-averaged".
    n_index: int    # 1,2,3... ordering of eigenstates for that (l,j) channel (or l-only if j2=None)
    energy_mev: float
    u: np.ndarray   # normalized u(r): ∫|u|^2 dr = 1
    occ: float      # occupation number


def fill_spherical_no_spin_orbit(
    tz: int,
    Nq: int,
    eigen_by_l: dict[int, list[tuple[float, np.ndarray]]],
) -> list[Orbital]:
    """
    Fill orbitals for one species (neutron/proton) with no spin-orbit:
      degeneracy g_l = 2 * (2l+1)  (spin included, but no j-splitting)

    eigen_by_l[l] = list of (E, u) sorted ascending.
    """
    # Flatten candidates with degeneracy
    candidates: list[tuple[float, int, int, np.ndarray, int]] = []
    for l, eigs in eigen_by_l.items():
        g = 2 * (2 * l + 1)
        for idx, (E, u) in enumerate(eigs, start=1):
            candidates.append((E, l, idx, u, g))

    candidates.sort(key=lambda x: x[0])

    occ_left = Nq
    orbitals: list[Orbital] = []
    for E, l, idx, u, g in candidates:
        if occ_left <= 0:
            break
        occ = float(min(g, occ_left))
        occ_left -= int(occ)
        orbitals.append(
            Orbital(tz=tz, l=l, j2=None, n_index=idx, energy_mev=float(E), u=u, occ=occ))

    if occ_left != 0:
        raise RuntimeError(f"Not enough states to fill tz={tz}: remaining {occ_left}")

    return orbitals

def fill_spherical_jj(
    tz: int,
    Nq: int,
    eigen_by_lj: dict[tuple[int, int], list[tuple[float, np.ndarray]]],
) -> list[Orbital]:
    """
    Fill orbitals for one species (neutron/proton) with spin-orbit splitting:
      each (l,j) channel has degeneracy g_j = 2j+1 = j2+1

    eigen_by_lj[(l, j2)] = list of (E, u) sorted ascending.
    """
    candidates: list[tuple[float, int, int, int, np.ndarray, int]] = []
    for (l, j2), eigs in eigen_by_lj.items():
        g = j2 + 1
        for idx, (E, u) in enumerate(eigs, start=1):
            candidates.append((E, l, j2, idx, u, g))

    candidates.sort(key=lambda x: x[0])

    occ_left = Nq
    orbitals: list[Orbital] = []
    for E, l, j2, idx, u, g in candidates:
        if occ_left <= 0:
            break
        occ = float(min(g, occ_left))
        occ_left -= int(occ)
        orbitals.append(
            Orbital(tz=tz, l=l, j2=j2, n_index=idx, energy_mev=float(E), u=u, occ=occ)
        )

    if occ_left != 0:
        raise RuntimeError(f"Not enough states to fill tz={tz}: remaining {occ_left}")

    return orbitals



def density_from_orbitals(mesh: RadialMesh, orbitals: list[Orbital]) -> np.ndarray:
    """
    Spherical density:
      rho(r) = sum_a occ_a * |R_a(r)|^2 / (4π)
    with u=rR => |R|^2 = u^2/r^2.

    So rho(r) = sum occ * u(r)^2 / (4π r^2)

    At r=0 handle with limit: u(0)=0 so safe; we set rho(0)=rho(1).
    """
    r = mesh.r
    rho = np.zeros_like(r)

    for orb in orbitals:
        u2 = orb.u * orb.u
        # avoid division by zero at r=0
        rho[1:] += orb.occ * (u2[1:] / (4.0 * np.pi * r[1:] ** 2))

    rho[0] = rho[1]
    return rho