"""
Continuum performance benchmarks.

Each module in this package is a stand-alone benchmark you can run via
``make bench-<name>``. Benchmarks emit ``bench/results/<name>.json``
plus a short stdout summary; see individual module docstrings.

The benchmarks deliberately *do not* import pytest fixtures — they are
runnable from a fresh checkout with no test infrastructure beyond the
pinned eval dependencies (``findings/longmemeval/repro/requirements-eval.txt``).
"""
