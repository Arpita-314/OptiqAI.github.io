# backend/decision_framework/decision_matrix.py
import numpy as np

class DecisionMatrix:
    def __init__(self, options, criteria, weights):
        """
        options: list of option names
        criteria: list of criteria names
        weights: list or array of weights (len == len(criteria))
        """
        assert len(criteria) == len(weights)
        self.options = options
        self.criteria = criteria
        self.weights = np.array(weights, dtype=float)
        self.scores = np.zeros((len(options), len(criteria)))

    def set_score(self, option_idx, crit_idx, val):
        self.scores[option_idx, crit_idx] = val

    def compute_weighted(self):
        weighted = (self.scores * self.weights).sum(axis=1) / max(self.weights.sum(), 1e-12)
        return weighted

    def recommend(self):
        weighted = self.compute_weighted()
        idx = int(np.argmax(weighted))
        return self.options[idx], float(weighted[idx]), weighted
