# backend/ai_engine/export_import.py
import json, os
from utils.logger import setup_logger

logger = setup_logger()

def export_knowledge(assistant, path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(assistant.knowledge_base, f, indent=2)
        logger.info(f"Exported assistant KB to {path}")
        return True
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return False

def import_knowledge(assistant, path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        for entry in kb:
            assistant.teach(entry.get("text",""))
        logger.info(f"Imported KB from {path}")
        return True
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return False
