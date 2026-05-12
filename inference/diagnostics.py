from __future__ import annotations

import numpy as np
from scipy import stats


def get_diagnostics(data: np.ndarray) -> dict[str, float]:
    """Summary statistics for the active numeric column (excess kurtosis from SciPy)."""
    data = np.asarray(data, dtype=float).ravel()
    n = int(data.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "skewness": float("nan"),
            "kurtosis": float("nan"),
        }
    return {
        "n": n,
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data, ddof=1)),
        "skewness": float(stats.skew(data, bias=False)),
        "kurtosis": float(stats.kurtosis(data, bias=False)),
    }


def is_binary_01(data: np.ndarray, atol: float = 1e-9) -> bool:
    """True if all values are in {0, 1} (within tolerance)."""
    data = np.asarray(data, dtype=float).ravel()
    if data.size == 0:
        return False
    u = np.unique(np.round(data, 6))
    return bool(np.all(np.isin(u, [0.0, 1.0])))
