from __future__ import annotations

import numpy as np

from src.skyrme import SkyrmeParams


def _couplings_C_rho(sk: SkyrmeParams) -> tuple[float, float, float, float]:
    # t0 part
    C0_rho = 3.0 / 8.0 * sk.t0
    C1_rho = -1.0 / 8.0 * sk.t0 * (2.0 * sk.x0 + 1.0)

    # t3 part
    C0_rho_a = 1.0 / 16.0 * sk.t3
    C1_rho_a = -1.0 / 48.0 * sk.t3 * (2.0 * sk.x3 + 1.0)

    return C0_rho, C1_rho, C0_rho_a, C1_rho_a


def central_mean_field_t0_t3(
    sk: SkyrmeParams,
    rho_n: np.ndarray,
    rho_p: np.ndarray,
    tz: int,
) -> np.ndarray:
    rho = rho_n + rho_p
    rho1 = rho_n - rho_p
    tau_q = 1.0 if tz == +1 else -1.0

    C0_rho, C1_rho, C0_rho_a, C1_rho_a = _couplings_C_rho(sk)

    U0 = 2.0 * C0_rho * rho + (sk.alpha + 2.0) * C0_rho_a * rho ** (sk.alpha + 1.0)
    U1 = 2.0 * C1_rho * rho1 + 2.0 * C1_rho_a * (rho ** sk.alpha) * rho1

    return U0 + tau_q * U1