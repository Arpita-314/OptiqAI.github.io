# backend/dataset_manager.py
import os, json
from utils.logger import setup_logger
from utils.config import DEFAULTS

logger = setup_logger()

class DatasetManager:
    def __init__(self, datasets_dir=None):
        self.datasets_dir = datasets_dir or DEFAULTS["datasets_dir"]
        os.makedirs(self.datasets_dir, exist_ok=True)
        self.index_file = os.path.join(self.datasets_dir, "index.json")
        if not os.path.exists(self.index_file):
            with open(self.index_file, "w") as f: json.dump({}, f)

    def register(self, name, path, meta=None):
        with open(self.index_file, "r") as f:
            idx = json.load(f)
        idx[name] = {"path": path, "meta": meta or {}}
        with open(self.index_file, "w") as f:
            json.dump(idx, f, indent=2)
        logger.info(f"Registered dataset {name}")

    def list_datasets(self):
        with open(self.index_file, "r") as f:
            return json.load(f)
