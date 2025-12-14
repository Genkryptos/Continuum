"""
Helpers for loading database configuration from ``config/db.yml``.

This module remains intentionally small so other components can import
``load_db_config`` without pulling in heavier dependencies.
"""

import yaml
from pathlib import Path

def load_db_config(env: str = "development"):
    """Return the DB config block for the requested environment."""
    config_path = Path(__file__).parent.parent / "config/db.yml"
    with open(config_path, "r") as f:
        all_cfg = yaml.safe_load(f)
    return all_cfg[env]

cfg = load_db_config()

print(cfg["URL"])
