# backend/optics_engine/optics_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from utils.logger import setup_logger

class OpticsSimulation:
    def __init__(self, name="Untitled Simulation"):
        self.logger = setup_logger()
        self.name = name
        self.results = None
        self.meta = {}

    def run_gaussian_beam(self, waist=1.0, x_range=5.0, N=500):
        self.logger.info(f"Running Gaussian beam: waist={waist}")
        x = np.linspace(-x_range, x_range, N)
        I = np.exp(-(x**2) / (2*(waist**2)))
        self.results = {"x": x.tolist(), "I": I.tolist()}
        self.meta = {"type":"gaussian", "waist": waist, "N": N}
        return self.results

    def visualize(self):
        if self.results is None:
            self.logger.warning("No results to visualize")
            return
        x = np.array(self.results["x"])
        I = np.array(self.results["I"])
        plt.figure(figsize=(6,3))
        plt.plot(x, I)
        plt.title(self.name)
        plt.xlabel("x")
        plt.ylabel("Intensity")
        plt.grid(True)
        plt.show()
