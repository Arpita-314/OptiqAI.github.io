# backend/ai_engine/assistant.py
import os, json
from utils.logger import setup_logger
from backend.ai_engine.model_manager import ModelManager
from utils.config import DEFAULTS

class OptiqAIAssistant:
    def __init__(self, model_type="local", model_name="baseline", store_path=None):
        self.logger = setup_logger()
        self.model_manager = ModelManager(model_type=model_type, model_name=model_name)
        self.model = self.model_manager.load_model()
        self.store_path = store_path or DEFAULTS["assistant_store"]
        self.knowledge_base = []
        self.load_store()
        self.logger.info("Assistant ready")

    def teach(self, text):
        entry = {"text": text}
        self.knowledge_base.append(entry)
        self.save_store()
        self.logger.info("Knowledge added")

    def ask(self, query):
        q = query.lower().strip()
        # naive retrieval: return first matching entry
        for entry in self.knowledge_base:
            if any(tok in entry["text"].lower() for tok in q.split()):
                return f"[{self.model['model_name']}] {entry['text'][:400]}"
        return f"[{self.model['model_name']}] Sorry — no matching knowledge. Teach me by clicking 'Teach'."

    def save_store(self):
        try:
            os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(self.knowledge_base, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save assistant store: {e}")

    def load_store(self):
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    self.knowledge_base = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load assistant store: {e}")
                self.knowledge_base = []
        else:
            self.knowledge_base = []
