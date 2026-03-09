from __future__ import annotations

import numpy as np

from src.skyrme import SkyrmeParams
from src.constants import HBAR2_OVER_2MN, HBAR2_OVER_2MP


def _hbar2_over_2m(tz: int) -> float:
    return HBAR2_OVER_2MN if tz == +1 else HBAR2_OVER_2MP


def B_field_t1t2(
    sk: SkyrmeParams,
    rho_n: np.ndarray,
    rho_p: np.ndarray,
    tz: int,
) -> np.ndarray:
    """
    Return B_q(r) in MeV*fm^2 for the kinetic operator:

        T_q = - d/dr [ B_q(r) d/dr ] + B_q(r) * l(l+1)/r^2   (in the u(r) equation)

    This is the minimal Skyrme effective-mass field contribution from t1/t2
    in the time-even, no-current limit.

    We use a standard decomposition:
      B_q(r) = (ħ²/2m_q) + C0^tau * rho(r) + C1^tau * rho_q(r)

    with:
      C0^tau = (1/16) * [ 3 t1 + t2 (5 + 4 x2) ]
      C1^tau = (1/16) * [ -t1 (1 + 2 x1) + t2 (1 + 2 x2) ]

    References: standard Skyrme EDF coupling constants (Bender/Heenen/Reinhard review).
    """
    rho = rho_n + rho_p
    rho_q = rho_n if tz == +1 else rho_p

    t1, t2, x1, x2 = sk.t1, sk.t2, sk.x1, sk.x2

    C0_tau = (1.0 / 16.0) * (3.0 * t1 + t2 * (5.0 + 4.0 * x2))
    C1_tau = (1.0 / 16.0) * (-t1 * (1.0 + 2.0 * x1) + t2 * (1.0 + 2.0 * x2))

    return _hbar2_over_2m(tz) + C0_tau * rho + C1_tau * rho_q