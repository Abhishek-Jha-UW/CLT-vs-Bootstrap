from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inference.diagnostics import get_diagnostics


@dataclass(frozen=True)
class MethodVerdict:
    """Human-readable guidance plus transparent checklist scoring."""

    headline: str
    detail: str
    clt_support_score: int
    max_score: int
    checks: dict[str, bool]


def compare_methods(data: np.ndarray, target_stat: str) -> MethodVerdict:
    """
    Lightweight heuristic: CLT-backed classical intervals are easiest to defend for the mean
    (t interval) and for binomial proportions (Wilson). Other targets are bootstrap-first.
    """
    data = np.asarray(data, dtype=float).ravel()
    diag = get_diagnostics(data)
    n = diag["n"]

    if target_stat in {"Median", "Std Dev", "75th Percentile"}:
        return MethodVerdict(
            headline="Prefer bootstrap for this estimand",
            detail=(
                "Classical closed-form intervals here either need strong parametric assumptions "
                "or specialized asymptotics. The percentile bootstrap targets the same quantity "
                "without assuming a particular population shape."
            ),
            clt_support_score=0,
            max_score=0,
            checks={},
        )

    if target_stat == "Proportion":
        if n >= 40 and abs(diag["skewness"]) < 0.75:
            return MethodVerdict(
                headline="Wilson and bootstrap should agree reasonably well",
                detail=(
                    "Wilson is a standard classical interval for binomial proportions. Bootstrap "
                    "of the sample mean (proportion) estimates the same parameter with fewer formulas."
                ),
                clt_support_score=2,
                max_score=2,
                checks={"n>=40": n >= 40, "|skew|<0.75 (binary skew)": abs(diag["skewness"]) < 0.75},
            )
        return MethodVerdict(
            headline="Wilson is still a strong default; compare to bootstrap",
            detail=(
                "With smaller n or proportions near 0/1, Wilson typically behaves better than a "
                "naive Wald interval. Bootstrap remains a useful cross-check."
            ),
            clt_support_score=1,
            max_score=2,
            checks={"n>=40": n >= 40, "|skew|<0.75": abs(diag["skewness"]) < 0.75},
        )

    # Mean on continuous-ish data: score CLT plausibility for the *mean* (not normality of raw data).
    score = 0
    checks: dict[str, bool] = {}
    checks["n >= 30"] = n >= 30
    checks["|skewness| < 0.75"] = abs(diag["skewness"]) < 0.75
    checks["excess kurtosis < 3"] = diag["kurtosis"] < 3
    score = sum(1 for v in checks.values() if v)

    if score >= 2:
        headline = "CLT/t interval is often reasonable; bootstrap is still a good check"
        detail = (
            "For the sample mean, the t interval is a standard frequentist default when sample "
            "size is moderate and tails are not extreme. Bootstrap helps when skew/heavy tails "
            "make asymptotic approximations shaky."
        )
    elif score == 1:
        headline = "Mixed evidence: prefer bootstrap or collect more data"
        detail = (
            "One CLT comfort signal is present, but others are weak. Bootstrap intervals are "
            "usually the safer default for communication unless assumptions are well understood."
        )
    else:
        headline = "Bootstrap is the safer default for this sample"
        detail = (
            "Small n, strong skew, and/or heavy tails weaken the usual CLT story for the *sampling "
            "distribution of the mean*. Bootstrap does not assume normality of the raw data."
        )

    return MethodVerdict(
        headline=headline,
        detail=detail,
        clt_support_score=score,
        max_score=len(checks),
        checks=checks,
    )
