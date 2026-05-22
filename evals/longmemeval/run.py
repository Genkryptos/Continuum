"""
evals/longmemeval/run.py
========================
CLI runner: wire :class:`ContinuumAdapter` into the LongMemEval harness
and persist the JSON results.

Usage
-----
.. code-block:: bash

    python -m evals.longmemeval.run \\
        --dataset longmemeval-s \\
        --answerer gpt-4o-mini \\
        --out results/

Or programmatically::

    from evals.longmemeval.run import run_benchmark
    results = await run_benchmark(
        dataset="longmemeval-s",
        answerer="gpt-4o-mini",
        adapter=my_adapter,
        benchmark_factory=LongMemEvalBenchmark,
    )

Design
------
``run_benchmark`` is *adapter- and benchmark-agnostic*: callers inject
both, so the CLI can wire a real LongMemEval install while tests pass
in fakes. The CLI itself just bridges argparse → ``run_benchmark`` and
writes the JSON.

Failures are logged and re-raised — a broken eval should fail loudly
rather than silently dumping a malformed JSON.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Factory signature for the LongMemEval harness. Defaults to lazy import.
BenchmarkFactory = Callable[..., Any]


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


async def run_benchmark(
    *,
    dataset: str,
    answerer: str,
    adapter: Any,
    benchmark_factory: BenchmarkFactory | None = None,
    out_dir: Path | str | None = None,
    results_filename: str | None = None,
    now: Callable[[], dt.datetime] | None = None,
) -> dict[str, Any]:
    """
    Run the LongMemEval benchmark with the given Continuum *adapter*.

    Parameters
    ----------
    dataset:
        Dataset name (e.g. ``"longmemeval-s"``, ``"longmemeval-l"``).
        Forwarded to the benchmark factory.
    answerer:
        Model name used by LongMemEval's scoring step (forwarded to
        the factory).
    adapter:
        The :class:`ContinuumAdapter` instance under test.
    benchmark_factory:
        Callable producing a benchmark object with ``async run()``
        returning a result mapping. Defaults to lazy-importing
        ``longmemeval.LongMemEvalBenchmark``.
    out_dir:
        Directory the JSON results file is written to. Defaults to
        ``./results``.
    results_filename:
        Override the auto-generated filename. The default is
        ``longmemeval_run_{ISO timestamp}.json``.
    now:
        Clock injection for deterministic test output.

    Returns
    -------
    The raw results mapping (also serialised to JSON on disk).
    """
    factory = benchmark_factory or _default_benchmark_factory
    benchmark = factory(dataset=dataset, adapter=adapter, answerer=answerer)

    log.info(
        "starting LongMemEval run dataset=%s answerer=%s", dataset, answerer
    )
    raw = await _maybe_await(benchmark.run())

    results = _normalise(raw)
    _log_summary(results)

    target = _resolve_results_path(out_dir, results_filename, now)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2, default=str))
    log.info("results written to %s", target)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_benchmark_factory(**kwargs: Any) -> Any:
    """
    Lazy import LongMemEval so this module loads even when the upstream
    repo isn't installed yet (CI imports it during smoke-tests).
    """
    try:
        from longmemeval import LongMemEvalBenchmark
    except ImportError as exc:  # pragma: no cover - exercised via setup.py
        raise ImportError(
            "LongMemEval is not installed. Run "
            "`python -m evals.longmemeval.setup` first."
        ) from exc
    return LongMemEvalBenchmark(**kwargs)


async def _maybe_await(value: Any) -> Any:
    """Accept both coroutine and plain-mapping ``run()`` return shapes."""
    if isinstance(value, Awaitable):
        return await value
    return value


def _normalise(raw: Any) -> dict[str, Any]:
    """
    LongMemEval's published shape gives ``accuracy`` / ``avg_tokens`` /
    ``total_cost``. Custom forks may use slightly different keys; fall
    back to the raw mapping so callers still get *something* useful.
    """
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(raw, "to_dict"):
        return dict(raw.to_dict())
    if hasattr(raw, "__dict__"):
        return dict(raw.__dict__)
    return {"raw": str(raw)}


def _log_summary(results: dict[str, Any]) -> None:
    acc = results.get("accuracy")
    avg = results.get("avg_tokens")
    cost = results.get("total_cost")
    if isinstance(acc, (int, float)):
        log.info("accuracy: %.1f%%", float(acc) * (100.0 if acc <= 1 else 1.0))
    if isinstance(avg, (int, float)):
        log.info("avg tokens per query: %.0f", float(avg))
    if isinstance(cost, (int, float)):
        log.info("total cost: $%.2f", float(cost))


def _resolve_results_path(
    out_dir: Path | str | None,
    filename: str | None,
    now: Callable[[], dt.datetime] | None,
) -> Path:
    base = Path(out_dir) if out_dir else Path("results")
    if filename is None:
        ts = (now or _utc_now)().strftime("%Y%m%dT%H%M%SZ")
        filename = f"longmemeval_run_{ts}.json"
    return base / filename


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


# ── CLI ────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LongMemEval against Continuum.")
    p.add_argument(
        "--dataset",
        default="longmemeval-s",
        help="LongMemEval dataset (longmemeval-s | longmemeval-l)",
    )
    p.add_argument(
        "--answerer",
        default="gpt-4o-mini",
        help="LLM model used by LongMemEval's scoring step",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results"),
        help="directory for results JSON",
    )
    return p.parse_args(argv)


def _cli_main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    """
    CLI entry — wires real Continuum + LongMemEval. Kept thin so the
    library entry ``run_benchmark`` stays test-friendly.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _parse_args(argv)  # validate flags even though _cli_main is a stub
    # The CLI deliberately defers adapter construction to a future task —
    # building a Continuum session needs configuration the user supplies
    # via ``continuum.yaml``. Once that lands wire it here:
    raise NotImplementedError(
        "Construct your ContinuumAdapter from your project's config and "
        "pass it to evals.longmemeval.run.run_benchmark() directly. "
        "See the module docstring for an example."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli_main())


__all__ = ["run_benchmark"]
