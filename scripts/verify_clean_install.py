"""
scripts/verify_clean_install.py
===============================
Verify that a freshly-installed continuum-memory wheel can import +
run a minimal :class:`ContinuumSession`. Called from
``make build-verify``; can also be run by hand inside a fresh venv:

::

    python -m venv /tmp/v && /tmp/v/bin/pip install dist/continuum_memory-*.whl
    /tmp/v/bin/python scripts/verify_clean_install.py
"""

from __future__ import annotations

import asyncio
import sys


def main() -> int:
    import continuum
    print(f"continuum.__version__ = {continuum.__version__}")

    from continuum.core.config import ContinuumConfig
    from continuum.core.session import ContinuumSession

    async def smoke() -> None:
        cfg = ContinuumConfig.load()
        async with ContinuumSession(cfg) as s:
            r = await s.process_turn("hello")
            print(f"reply: {r[:80]!r}")
            print(f"session_id: {s.session_id}")

    asyncio.run(smoke())
    print("CLEAN INSTALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
