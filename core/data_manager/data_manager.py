import yaml, os, json
from datetime import datetime

class DataManager:
    """Manages datasets, experiment configs, and results."""
    def __init__(self, base_dir="experiments"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_experiment(self, name, data):
        path = os.path.join(self.base_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[OptiqAI] Experiment saved to {path}")

    def list_experiments(self):
        return [f for f in os.listdir(self.base_dir) if f.endswith(".json")]
