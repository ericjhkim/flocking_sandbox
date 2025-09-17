"""
This is double-integrator dynamics for a Laplacian-Pinning control law (control stable) augmented by a mass-spring system.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tools import plots, agents

# Controls
gif_path = "visualizations/laplacian_pinning_spring/"
CREATE_GIF = False
SEED = 1

# Constants
SIM_TIME = 10                               # Simulation time in seconds
N_AGENTS = 8                                # Number of agents

k1 = 1.0                                    # Relative formation control gain
k2 = 2.0                                    # Absolute position control gain
k3 = 3.0                                    # Velocity damping gain

def main():
    # Set pin(s)
    pins = np.arange(0, N_AGENTS)

    # Create flock
    flock = Flock(2, N_AGENTS, [k1, k2, k3], pins, SEED)

    # Run simulation
    for t in np.arange(0, SIM_TIME, flock.dt):
        flock.update(t)

    plots.animate_3d(flock, CREATE_GIF, gif_path)

class Flock(agents.Agents):
    def __init__(self, N_INT, N_AGENTS, gains, pins, SEED):
        super().__init__(N_INT, True, N_AGENTS, SEED, T_VEC=[20,20,0])

        # Control gains
        self.k1, self.k2, self.k3 = gains

        # Pins and global position
        self.pins = pins
        self.P = np.diag(np.zeros(self.N_AGENTS))                   # Pinning matrix
        self.P[pins,pins] = 1

        # Spring model
        self.L1 = np.zeros((self.N_AGENTS, self.N_AGENTS))          # Natural spring length matrix
        for i in range(self.N_AGENTS):
            for j in range(self.N_AGENTS):
                self.L1[i, j] = np.linalg.norm(self.X_tgt[i] - self.X_tgt[j])
        self.K1 = self.A_T1                                         # Spring constant matrix

    def control(self, t):
        # Update adjacency matrix
        # self.A1 = self.get_adjacency(self.X)
        
        # Compute spring law
        U = np.zeros((self.N_AGENTS, 3))
        for i in range(self.N_AGENTS):
            for j in range(self.N_AGENTS):
                if self.A1[i, j] != 0:  # If neighbours
                    U[i] += self.K1[i,j]*(np.linalg.norm(self.X[j] - self.X[i]) - self.L1[i,j]) * (self.X[j] - self.X[i])/np.linalg.norm(self.X[j] - self.X[i])
            U[i] -= self.k3*self.V[i]

        # Compute laplacian matrix
        L = self.compute_laplacian(self.A1)

        # Compute formation control
        error = self.X - self.X_tgt                                 # Error in position
        U += -self.k1 * L @ error - self.k2 * self.P @ error

        return U

if __name__ == "__main__":
    main()