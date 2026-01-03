# backend/experiment_manager.py
from utils.logger import get_logger
from utils.env_capture import capture_env

class ExperimentManager:
    def __init__(self, name):
        self.name = name
        self.logger = get_logger(name)
        self.env_info = capture_env()
    
    def start_run(self, config):
        self.logger.info(f"Starting run {self.name}")
        # Save config + env info
        # MLflow or local JSON logs here
