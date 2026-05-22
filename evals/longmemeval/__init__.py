"""
evals.longmemeval
=================
LongMemEval benchmark harness for Continuum.

Three entry points:

* :mod:`evals.longmemeval.setup`    — clone + install LongMemEval.
* :mod:`evals.longmemeval.adapter`  — :class:`ContinuumAdapter` mapping
  Continuum's session/retrieve loop onto LongMemEval's expected
  ``process_conversation`` / ``answer_question`` interface.
* :mod:`evals.longmemeval.run`      — CLI runner that loads config, wires
  the adapter, runs the benchmark, persists JSON results.

Reference: https://github.com/xiaowu0162/LongMemEval
"""
from __future__ import annotations

from evals.longmemeval.adapter import ContinuumAdapter
from evals.longmemeval.baseline import (
    BaselineResults,
    EvalRow,
    RowResult,
    default_scorer,
    run_baseline,
)
from evals.longmemeval.run import run_benchmark
from evals.longmemeval.setup import setup_longmemeval

__all__ = [
    "ContinuumAdapter",
    "run_benchmark",
    "setup_longmemeval",
    "run_baseline",
    "BaselineResults",
    "EvalRow",
    "RowResult",
    "default_scorer",
]
