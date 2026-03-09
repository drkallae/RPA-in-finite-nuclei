from __future__ import annotations

from src.mesh import RadialMesh
from src.hf.potentials import woods_saxon
from src.hf.radial_solve import solve_radial_bound_states_box


def main() -> None:
    mesh = RadialMesh.uniform(r_max_fm=20.0, n_points=2000)

    A = 16
    R0 = 1.2 * (A ** (1.0 / 3.0))
    U_n = woods_saxon(mesh, V0_mev=50.0, R0_fm=R0, a_fm=0.65)

    for l in [0, 1, 2]:
        states = solve_radial_bound_states_box(mesh, U_mev=U_n, l=l, tz=+1, n_states=4)
        print(f"l={l} (neutron), lowest energies:")
        for i, (E, _u) in enumerate(states, start=1):
            print(f"  state {i}: E = {E:9.4f} MeV")
        print()


if __name__ == "__main__":
    main()