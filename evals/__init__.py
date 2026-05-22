"""
continuum.evals
===============
Benchmark harnesses for evaluating Continuum against published memory
benchmarks. Each sub-package wraps a specific benchmark behind a small
adapter so the framework code can stay clean of benchmark-specific shims.

Currently implemented
---------------------
* :mod:`evals.longmemeval` — wraps the LongMemEval benchmark
  (https://github.com/xiaowu0162/LongMemEval).
"""
from __future__ import annotations
