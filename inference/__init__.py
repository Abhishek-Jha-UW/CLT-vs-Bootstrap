"""Inference toolkit: diagnostics, classical intervals, bootstrap, and optional LLM narration."""

from inference.bootstrap import bootstrap_ci
from inference.clt import clt_mean_t_ci, wilson_proportion_ci
from inference.compare import compare_methods, MethodVerdict
from inference.diagnostics import get_diagnostics, is_binary_01
from inference.insights import get_ai_insights

__all__ = [
    "bootstrap_ci",
    "clt_mean_t_ci",
    "wilson_proportion_ci",
    "compare_methods",
    "MethodVerdict",
    "get_diagnostics",
    "is_binary_01",
    "get_ai_insights",
]
