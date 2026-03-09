from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Nucleus:
    Z: int
    N: int

    @property
    def A(self) -> int:
        return self.Z + self.N


@dataclass(frozen=True)
class MeshConfig:
    r_max_fm: float = 20.0
    n_points: int = 2000  # dense enough for later; can reduce during debugging


@dataclass(frozen=True)
class HFConfig:
    max_iter: int = 200
    mixing: float = 0.3
    tol_energy: float = 1e-6  # MeV, placeholder (you'll define energy calc later)
    tol_density: float = 1e-6  # placeholder


@dataclass(frozen=True)
class RunConfig:
    nucleus: Nucleus
    mesh: MeshConfig
    hf: HFConfig
    skyrme_name: str = "SLy4"
    output_dir: Path = Path("out/week1_o16_sly4_e2")
    channel: str = "E2"  # for later SRPA/RPA stage