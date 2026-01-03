import numpy as np

class MatrixTransformations:
    """Performs optical matrix operations used in lenses, mirrors, and systems."""

    def __init__(self):
        pass

    def translation(self, d):
        """Free-space propagation by distance d."""
        return np.array([[1, d], [0, 1]])

    def thin_lens(self, f):
        """Thin lens matrix with focal length f."""
        return np.array([[1, 0], [-1/f, 1]])

    def reflection(self):
        """Mirror reflection matrix."""
        return np.array([[1, 0], [0, -1]])

    def system_matrix(self, matrices):
        """Combine multiple matrices in sequence."""
        result = np.eye(2)
        for M in matrices:
            result = M @ result
        return result

    def beam_propagation(self, y_in, theta_in, system):
        """Propagate a ray through system matrix."""
        input_vec = np.array([[y_in], [theta_in]])
        output = system @ input_vec
        return output[0,0], output[1,0]

if __name__ == "__main__":
    mt = MatrixTransformations()
    lens = mt.thin_lens(10)
    free = mt.translation(5)
    system = mt.system_matrix([lens, free])
    print("System matrix:\n", system)
    y_out, th_out = mt.beam_propagation(1, 0.1, system)
    print("Output ray:", y_out, th_out)
