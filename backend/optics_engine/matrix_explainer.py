# backend/optics_engine/matrix_explainer.py
import json, os
from utils.logger import setup_logger
from utils.config import DEFAULTS

class MatrixExplainer:
    def __init__(self, lib_path=None):
        self.logger = setup_logger()
        self.lib_path = lib_path or DEFAULTS["matrix_library"]
        if os.path.exists(self.lib_path):
            with open(self.lib_path, "r", encoding="utf-8") as f:
                self.lib = json.load(f)
        else:
            self.lib = {}
        self.logger.info("MatrixExplainer loaded")

    def explain(self, key):
        return self.lib.get(key, {"error":"Unknown matrix type"})

    def detect_by_shape(self, arr):
        # naive detection by shape and content
        import numpy as np
        a = np.asarray(arr)
        if a.shape == (2,2):
            # heuristic: if floats and small -> ABCD or Jones candidate
            return "ABCD" if a.dtype.kind in 'f' else "ABCD"
        if a.shape == (4,4):
            return "Mueller"
        if a.ndim == 2 and a.shape[0] == a.shape[1]:
            return "Fourier" if np.iscomplexobj(a) or a.dtype.kind in 'f' else "Unknown"
        return "Unknown"
