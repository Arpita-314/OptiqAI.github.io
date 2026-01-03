# backend/optics_engine/matrix_transformer.py
import numpy as np
from utils.logger import setup_logger

class MatrixTransformer:
    def __init__(self):
        self.logger = setup_logger()
        self.logger.info("MatrixTransformer ready")

    def abcd_free_space(self, L):
        M = np.array([[1.0, L],
                      [0.0, 1.0]])
        self.logger.debug(f"ABCD free space L={L}\n{M}")
        return M

    def abcd_thin_lens(self, f):
        M = np.array([[1.0, 0.0],
                      [-1.0/f, 1.0]])
        self.logger.debug(f"ABCD thin lens f={f}\n{M}")
        return M

    def apply_abcd(self, M, ray):
        # ray is [y; theta] column vector
        ray = np.asarray(ray).reshape(2,1)
        out = M @ ray
        return out.flatten()

    # generic affine transforms (2x2 / 3x3)
    def rotate(self, angle_deg):
        theta = np.deg2rad(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        self.logger.debug(f"Rotation {angle_deg} deg\n{R}")
        return R

    def translate_homogeneous(self, tx, ty):
        T = np.array([[1, 0, tx],
                      [0, 1, ty],
                      [0, 0, 1]])
        return T
