# potentials.py
import numpy as np

# Base potential interface
class Potential:
    def __init__(self, **params):
        self.params = params
        self.eps = 1e-6  # Small constant to avoid division by zero

    def compute(self, x_i, x_j, v_i, v_j):
        raise NotImplementedError("Must implement compute() method")

# Lennard-Jones
class LennardJones(Potential):
    def compute(self, x_i, x_j, v_i, v_j):
        r = np.linalg.norm(x_i - x_j)

        a = self.params.get('a', 0.2)
        b = self.params.get('b', 1.0)
        r0 = self.params.get('r0', 1.0)
        eps = self.eps

        p = (-12*a/((r - r0)**13 + eps) + 6*b/((r - r0)**7 + eps)) * (x_i - x_j)/(r + eps) - 0.5*(v_i - v_j)
        return p

# Quadratic Potential
class Quadratic(Potential):
    def compute(self, x_i, x_j, v_i, v_j):
        r = np.linalg.norm(x_i - x_j)

        k = self.params.get('k', 1.0)
        r0 = self.params.get('r0', 1.0)
        eps = self.eps

        p = -k * (r - r0) * (x_i - x_j)/(r + eps) - 0.5*(v_i - v_j)
        return p
    
class Morse(Potential):
    def compute(self, x_i, x_j, v_i, v_j):
        r = np.linalg.norm(x_i - x_j)

        D = self.params.get('D', 1.0)      # well depth
        a = self.params.get('a', 1.0)      # stiffness
        r0 = self.params.get('r0', 1.0)    # equilibrium distance
        eps = self.eps

        p = 2*a*D*(1 - np.exp(-a*(r - r0))) * np.exp(-a*(r - r0)) * (x_i - x_j)/(r + eps) - 0.5*(v_i - v_j)
        return p

# Potential registry for easy selection
POTENTIALS = {
    "lennard_jones": LennardJones,
    "quadratic": Quadratic,
    "morse": Morse,
}