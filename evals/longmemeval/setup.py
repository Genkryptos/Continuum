"""
evals/longmemeval/setup.py
==========================
One-shot setup script: clone the upstream LongMemEval repository and
install its Python dependencies into the current environment.

Usage
-----
.. code-block:: bash

    cd evals/longmemeval
    python setup.py              # idempotent — re-running is safe
    python setup.py --force      # delete + reclone

Design
------
* Idempotent: a second invocation runs ``git pull`` instead of failing
  on an existing clone.
* Network calls are isolated in helper functions so unit tests can
  patch :mod:`subprocess` cleanly.
* Returns the absolute path to the cloned directory so downstream
  scripts can locate the dataset without env-var gymnastics.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/xiaowu0162/LongMemEval.git"
DEFAULT_TARGET = Path(__file__).parent / "LongMemEval"


def setup_longmemeval(
    *,
    target: Path | str | None = None,
    force: bool = False,
    runner: object | None = None,
) -> Path:
    """
    Clone LongMemEval and install its requirements.

    Parameters
    ----------
    target:
        Directory the repo is cloned into. Defaults to
        ``evals/longmemeval/LongMemEval`` next to this file.
    force:
        Delete *target* first if it already exists.
    runner:
        A ``subprocess``-compatible object — must expose ``run(args,
        check=True)``. Defaults to the real :mod:`subprocess` module.
        Tests inject a fake.

    Returns
    -------
    Path to the cloned directory.
    """
    target_path = Path(target) if target else DEFAULT_TARGET
    runner = runner if runner is not None else subprocess

    if target_path.exists() and force:
        log.info("removing existing clone at %s", target_path)
        shutil.rmtree(target_path)

    if target_path.exists():
        log.info("LongMemEval already cloned — fetching latest")
        _run(runner, ["git", "-C", str(target_path), "pull", "--ff-only"])
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("cloning %s → %s", REPO_URL, target_path)
        _run(runner, ["git", "clone", REPO_URL, str(target_path)])

    reqs = target_path / "requirements.txt"
    if reqs.exists():
        log.info("installing LongMemEval dependencies from %s", reqs)
        _run(
            runner,
            [sys.executable, "-m", "pip", "install", "-r", str(reqs)],
        )
    else:
        log.warning(
            "no requirements.txt in %s — skipping pip install step",
            target_path,
        )
    return target_path


def _run(runner: object, args: list[str]) -> None:
    """Call ``runner.run(args, check=True)``; surfaces failures clearly."""
    try:
        runner.run(args, check=True)  # type: ignore[attr-defined]
    except subprocess.CalledProcessError as exc:
        log.error("command failed (%d): %s", exc.returncode, " ".join(args))
        raise


# ── CLI ────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Set up LongMemEval for Continuum.")
    p.add_argument(
        "--target",
        type=Path,
        default=None,
        help="directory to clone into (default: ./LongMemEval)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="delete an existing clone before re-cloning",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    setup_longmemeval(target=args.target, force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())


__all__ = ["setup_longmemeval", "REPO_URL", "DEFAULT_TARGET"]
