"""
This is single-integrator dynamics for a Laplacian-Pinning control law (control stable).

@author: Kim.ej
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tools import plots, agents
import random

# Controls
gif_path = "visualizations/laplacian_pinning/"
CREATE_GIF = False
SEED = 1
np.random.seed(SEED)

# Constants
SIM_TIME = 10                               # Simulation time in seconds
N_AGENTS = 8                                # Number of agents

k1 = 1.0                                    # Relative formation control gain
k2 = 1.0                                    # Absolute position control gain

def main():
    # Set pin(s)
    pins = np.arange(0, N_AGENTS)

    # Create flock
    flock = Flock(1, N_AGENTS, [k1, k2], pins, SEED=SEED)

    # Run simulation
    for t in np.arange(0, SIM_TIME, flock.dt):
        flock.update(t)

    plots.animate_3d(flock, CREATE_GIF, gif_path)

class Flock(agents.SubgraphAgents):
    def __init__(self, N_INT, N_AGENTS, gains, pins, SEED=random.seed()):
        super().__init__(N_INT, N_AGENTS, SEED)

        # Constants
        self.MAX_VEL = 10.0                                         # Maximum velocity of agents

        # Control gains
        self.k1, self.k2 = gains

        # Pins and global position
        self.pins = pins
        self.P = np.diag(np.zeros(self.N_AGENTS))                   # Pinning matrix
        self.P[pins,pins] = 1

    def control(self):
        # Compute laplacian
        L = self.compute_laplacian(self.get_adjacency(self.X))

        # Compute formation control
        error = self.X - self.X_tgt                                 # Error in position
        U = -self.k1 * L @ error - self.k2 * self.P @ error

        # Apply velocity saturation
        speed = np.linalg.norm(U, axis=1)
        speed_clipped = np.minimum(speed, self.MAX_VEL)             # Limit speeds
        U = (U.T * (speed_clipped / (speed + 1e-6))).T              # Normalize and scale

        return U

if __name__ == "__main__":
    main()