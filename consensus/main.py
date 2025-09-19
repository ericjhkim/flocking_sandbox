"""
Showcase of convergence to consensus depending on graph connectivity (Fiedler value).
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tools import agents
import plots

# Controls
gif_path = "visualizations/consensus/"
CREATE_GIF = False
SEED = 2

# Constants
SIM_TIME = 5                                # Simulation time in seconds
N_AGENTS = 3                                # Number of agents

def main():
    # Create flock
    flock1 = Flock(2, N_AGENTS, SEED, R_MAX=6.0)
    flock2 = Flock(2, N_AGENTS, SEED, R_MAX=5.5)

    # Run simulation
    for t in np.arange(0, SIM_TIME, flock1.dt):
        flock1.update(t)
        flock2.update(t)

    plots.animate_3d([flock1, flock2], CREATE_GIF, gif_path)

class Flock(agents.Agents):
    def __init__(self, N_INT, N_AGENTS, SEED, R_MAX=6.0):
        super().__init__(N_INT, False, N_AGENTS, SEED, D_MIN=5.0, D_MAX=6.0, R_MAX=R_MAX)

        print(f"Flock Laplacian: {self.compute_laplacian(self.get_adjacency(self.X))}")
        print(f"Flock eigenvalues: {np.linalg.eigvals(self.compute_laplacian(self.get_adjacency(self.X)))}")

    def control(self, t):
        # Update adjacency matrix
        self.A1 = self.get_adjacency(self.X)

        U = np.zeros((self.N_AGENTS, 3))

        for i in range(N_AGENTS):
            q_i = self.X[i]
            p_i = self.V[i]
            for j in range(N_AGENTS):
                if i != j and self.A1[i,j] == 1: # Agent i is neighbour of agent j
                    q_j = self.X[j]
                    U[i] -= (q_i - q_j)

            U[i] -= 3.0*p_i

        return U

if __name__ == "__main__":
    main()