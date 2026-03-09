from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkyrmeParams:
    # Standard Skyrme parameters (MeV, fm units in conventional Skyrme form)
    t0: float
    t1: float
    t2: float
    t3: float
    x0: float
    x1: float
    x2: float
    x3: float
    alpha: float
    W0: float  # spin-orbit strength


def sly4() -> SkyrmeParams:
    """
    SLy4 parameterization (common values).
    Note: in Week 1 we store parameters; Week 2 will use them in mean fields.
    """
    # Widely used published SLy4 set:
    # t0=-2488.91, t1=486.82, t2=-546.39, t3=13777.0,
    # x0=0.834, x1=-0.344, x2=-1.0, x3=1.354, alpha=1/6, W0=123.0
    return SkyrmeParams(
        t0=-2488.91,
        t1=486.82,
        t2=-546.39,
        t3=13777.0,
        x0=0.834,
        x1=-0.344,
        x2=-1.0,
        x3=1.354,
        alpha=1.0 / 6.0,
        W0=123.0,
    )


def get_skyrme(name: str) -> SkyrmeParams:
    key = name.strip().upper()
    if key in {"SLY4", "SLY4*", "SLY4STAR", "SLY4STAR*"}:
        # We'll treat SLy4* same as SLy4 for now unless you provide exact SLy4* numbers.
        return sly4()
    raise ValueError(f"Unknown Skyrme set: {name}")
