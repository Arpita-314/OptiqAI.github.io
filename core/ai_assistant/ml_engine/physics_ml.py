import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

class PhysicsInformedML:
    """Simple ML module for regression on optics data."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()

    def train(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        print("[OptiqAI] Model trained successfully.")

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)

    def save(self, path="ml_model.pkl"):
        joblib.dump((self.scaler, self.model), path)
        print("[OptiqAI] Model saved.")

    def load(self, path="ml_model.pkl"):
        self.scaler, self.model = joblib.load(path)
        print("[OptiqAI] Model loaded.")
