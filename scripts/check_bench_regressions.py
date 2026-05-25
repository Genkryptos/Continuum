"""
scripts/check_bench_regressions.py
==================================
Gate the four Phase-3B benchmarks against the contract thresholds the
report claims. Reads the latest JSON outputs from ``bench/results/``
and exits ``0`` if every threshold holds, ``1`` otherwise.

Thresholds (matching the README headline table)::

    supersession_correctness.continuum_supersession  ≥ 95 %
    bi_temporal.continuum_bitemporal                 = 100 % (all 20)
    retrieval_quality.continuum_stm                  ≥ naive_cosine recall@4 − 5pp
    ingest_throughput.continuum_full                 available + finite

Called from ``.github/workflows/benchmarks.yml`` as the regression
guard, and runnable locally after ``make bench-all`` to confirm a
local change hasn't broken anything important.

Usage::

    python scripts/check_bench_regressions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "bench" / "results"


def _load(name: str) -> dict | None:
    """Return the parsed JSON at ``bench/results/<name>_latest.json`` or None."""
    path = RESULTS / f"{name}_latest.json"
    if not path.exists():
        return None
    # The "latest" entries are symlinks; if the symlink is dangling
    # treat that as the file being missing.
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _by_name(data: dict, system: str) -> dict | None:
    for s in data.get("systems", []):
        if s.get("system") == system:
            return s
    return None


def check_supersession() -> tuple[bool, str]:
    d = _load("supersession")
    if d is None:
        return False, "supersession_latest.json missing — did `make bench-supersession` run?"
    sys_ = _by_name(d, "continuum_supersession")
    if sys_ is None:
        return False, "continuum_supersession entry missing from supersession results"
    pct = float(sys_.get("correctness_pct", 0))
    bar = 95.0
    ok = pct >= bar
    return ok, f"supersession: continuum_supersession = {pct:.1f}% (bar {bar}%)"


def check_bi_temporal() -> tuple[bool, str]:
    d = _load("bi_temporal")
    if d is None:
        return False, "bi_temporal_latest.json missing — did `make bench-bitemporal` run?"
    sys_ = _by_name(d, "continuum_bitemporal")
    if sys_ is None:
        return False, "continuum_bitemporal entry missing from bi_temporal results"
    n = int(sys_.get("n_total", 0))
    correct = int(sys_.get("n_correct", 0))
    ok = correct == n and n >= 20
    return ok, f"bi_temporal: continuum_bitemporal = {correct}/{n} (need 20/20)"


def check_retrieval_quality() -> tuple[bool, str]:
    d = _load("retrieval")
    if d is None:
        return False, "retrieval_latest.json missing — did `make bench-retrieval` run?"
    naive = _by_name(d, "naive_cosine") or {}
    cont = _by_name(d, "continuum_stm") or {}
    if not naive or not cont:
        return False, "retrieval: expected naive_cosine + continuum_stm rows in results"
    naive_r4 = float(naive.get("recall_at_4", 0)) * 100
    cont_r4 = float(cont.get("recall_at_4", 0)) * 100
    # Don't require a beat — just no significant regression vs the
    # raw-cosine baseline (this corpus has no temporal signal for the
    # scorer to leverage, so parity is the honest contract).
    ok = cont_r4 >= naive_r4 - 5.0
    return ok, (
        f"retrieval: continuum_stm r@4 = {cont_r4:.1f}%, "
        f"naive_cosine r@4 = {naive_r4:.1f}% (parity tolerance ±5pp)"
    )


def check_ingest_throughput() -> tuple[bool, str]:
    d = _load("ingest")
    if d is None:
        return False, "ingest_latest.json missing — did `make bench-ingest` run?"
    cont = _by_name(d, "continuum_full") or {}
    if not cont:
        return False, "ingest: continuum_full entry missing"
    if not cont.get("available", False):
        return False, f"ingest: continuum_full unavailable — note: {cont.get('note','')[:80]}"
    # Sanity: per-session p50 should be a finite positive number.
    p50 = float(cont.get("p50_ms_per_session", -1))
    ok = p50 >= 0 and p50 < 5000   # 5s is absurd ceiling for in-memory
    return ok, (
        f"ingest: continuum_full p50 = {p50:.2f} ms/session, "
        f"{cont.get('llm_calls_per_session',0):.1f} LLM calls/session"
    )


CHECKS = [
    ("supersession", check_supersession),
    ("bi_temporal",  check_bi_temporal),
    ("retrieval",    check_retrieval_quality),
    ("ingest",       check_ingest_throughput),
]


def main() -> int:
    print("=" * 72)
    print("  Phase-3B benchmark regression gate")
    print("=" * 72)
    fails = 0
    for _name, fn in CHECKS:
        ok, msg = fn()
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {msg}")
        if not ok:
            fails += 1
    print("=" * 72)
    if fails:
        print(f"  ❌ {fails} regression(s)")
        return 1
    print("  ✅ all benchmarks within contract thresholds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
