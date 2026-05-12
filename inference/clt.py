from __future__ import annotations

import numpy as np
from scipy import stats


def clt_mean_t_ci(data: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided CI for the population mean using the Student t pivot (CLT + finite-sample correction)."""
    data = np.asarray(data, dtype=float).ravel()
    n = data.size
    if n < 2:
        raise ValueError("Need at least two observations for a t-based mean CI.")
    mean = float(np.mean(data))
    se = float(stats.sem(data))
    df = n - 1
    t_crit = float(stats.t.ppf((1 + confidence) / 2, df=df))
    margin = t_crit * se
    return mean - margin, mean + margin


def wilson_proportion_ci(data: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion (mean of Bernoulli trials).
    More stable than a naive normal (Wald) interval near 0 or 1.
    """
    data = np.asarray(data, dtype=float).ravel()
    n = int(data.size)
    if n == 0:
        raise ValueError("Empty data.")
    x = int(np.sum(data))
    z = float(stats.norm.ppf((1 + confidence) / 2))
    p = x / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(max(p * (1 - p) / n + z**2 / (4 * n**2), 0.0)) / denom
    return float(centre - half), float(centre + half)
