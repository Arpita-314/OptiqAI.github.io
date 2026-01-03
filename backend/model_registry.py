# backend/model_registry.py
import json, os
from datetime import datetime
from utils.logger import setup_logger
from utils.config import DEFAULTS

logger = setup_logger()

class ModelRegistry:
    def __init__(self, registry_dir=None):
        self.dir = registry_dir or DEFAULTS["models_dir"]
        os.makedirs(self.dir, exist_ok=True)
        self.manifest = os.path.join(self.dir, "manifest.json")
        if not os.path.exists(self.manifest):
            with open(self.manifest, "w") as f: json.dump({}, f)

    def register(self, model_id, meta: dict):
        with open(self.manifest, "r") as f:
            d = json.load(f)
        d[model_id] = meta
        with open(self.manifest, "w") as f:
            json.dump(d, f, indent=2)
        logger.info(f"Registered model {model_id}")

    def list_models(self):
        with open(self.manifest, "r") as f:
            return json.load(f)
