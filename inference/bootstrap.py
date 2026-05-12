from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    stat_func: Callable[..., np.ndarray] = np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 5000,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[float, float], np.ndarray]:
    """
    Percentile bootstrap CI for any statistic that can be reduced along axis=1.

    Returns ((lower, upper), bootstrap_stat_samples)).
    """
    data = np.asarray(data, dtype=float).ravel()
    if data.size == 0:
        raise ValueError("Empty data.")
    if rng is None:
        rng = np.random.default_rng()
    resamples = rng.choice(data, size=(n_bootstrap, data.size), replace=True)
    boot_samples = stat_func(resamples, axis=1)
    boot_samples = np.asarray(boot_samples, dtype=float).ravel()
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(boot_samples, alpha * 100))
    upper = float(np.percentile(boot_samples, (1 - alpha) * 100))
    return (lower, upper), boot_samples
