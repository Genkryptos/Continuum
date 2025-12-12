import yaml
from pathlib import Path

def load_db_config(env: str = "development"):
    config_path = Path(__file__).parent.parent / "config/db.yml"
    with open(config_path, "r") as f:
        all_cfg = yaml.safe_load(f)
    return all_cfg[env]

cfg = load_db_config()

print(cfg["URL"])