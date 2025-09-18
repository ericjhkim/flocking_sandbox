"""
This is a sandbox for exploring different potential functions.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tools import agents
import plots
from potentials import POTENTIALS
from init_conditions import INIT_FUNCS

# Controls
gif_path = "visualizations/potentials/quadratic_"
CREATE_GIF = False
SEED = np.random.randint(0, 1e6)
SEED = 3
print(SEED)

# Constants
SIM_TIME = 5                               # Simulation time in seconds
N_AGENTS = 2                                # Number of agents

potential_name = "quadratic"

def main():
    init_func = INIT_FUNCS[potential_name]
    X, V, params = init_func(N_AGENTS, SEED, min_dist=3, max_dist=3.5)

    potential_class = POTENTIALS[potential_name]
    potential_obj = potential_class(**params)

    flock = Flock(2, N_AGENTS, SEED, potential_obj, X, V)

    # Run simulation
    for t in np.arange(0, SIM_TIME, flock.dt):
        flock.update(t)

    plots.animate_3d(flock, CREATE_GIF, gif_path, follow=True, follow_padding=0.5)

class Flock(agents.Agents):
    def __init__(self, N_INT, N_AGENTS, SEED, potential_obj, X, V):
        super().__init__(N_INT, True, N_AGENTS, SEED, VERBOSE=False)
        self.potential = potential_obj

        self.X = X
        self.V = V
        self.data['position'] = [self.X.copy()]
        self.data['velocity'] = [self.V.copy()]

        delattr(self, "X_tgt")

    def control(self, t):
        self.A1 = self.get_adjacency(self.X)
        U = np.zeros((self.N_AGENTS, 3))
        for i in range(self.N_AGENTS):
            for j in range(self.N_AGENTS):
                if self.A1[i, j] != 0:
                    u_i = self.potential.compute(self.X[i], self.X[j], self.V[i], self.V[j])
                    U[i] += u_i
        return U
    
if __name__ == "__main__":
    main()