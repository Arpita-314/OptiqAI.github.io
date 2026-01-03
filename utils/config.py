# utils/config.py
import os, json
ROOT = os.path.dirname(os.path.dirname(__file__)) if __name__ != "__main__" else os.getcwd()
DEFAULTS = {
    "runs_dir": os.path.join(ROOT, "runs"),
    "models_dir": os.path.join(ROOT, "models"),
    "datasets_dir": os.path.join(ROOT, "data"),
    "assistant_store": os.path.join(ROOT, "data", "assistant_memory.json"),
    "matrix_library": os.path.join(ROOT, "backend", "optics_engine", "matrix_library.json")
}
os.makedirs(DEFAULTS["runs_dir"], exist_ok=True)
os.makedirs(DEFAULTS["models_dir"], exist_ok=True)
os.makedirs(DEFAULTS["datasets_dir"], exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
