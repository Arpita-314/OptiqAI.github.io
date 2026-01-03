# backend/experiment_logger.py
import os, json, hashlib
from datetime import datetime
from utils.logger import setup_logger
from utils.config import DEFAULTS

logger = setup_logger()

def make_run_dir(base=None, prefix="run"):
    base = base or DEFAULTS["runs_dir"]
    os.makedirs(base, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    run_id = f"{prefix}_{ts}"
    path = os.path.join(base, run_id)
    os.makedirs(path, exist_ok=True)
    return path

def save_run_meta(run_dir, meta):
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved run meta to {meta_path}")
    return meta_path
