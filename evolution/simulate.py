"""
Simulate NEAT-evolved ANN.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle
import neat
from tools import plots
from evolve import Flock
import evolution.neat_tools as neat_tools

# Loading parameters
files = [x for x in os.listdir("evolution/networks/") if "." not in x and x != "neat_config_tmp"]
net_id = sorted(files)[-1]
print(f"Loaded network {net_id}")

# Loading data
with open(f'D:\\EK Research\\flocking_sandbox\\evolution\\networks\\{net_id}', 'rb') as f:
    data = pickle.load(f)
    genome = data["genome"]
    settings = data["settings"]
    print(data["description"])

# Visualization settings
gif_path = "visualizations/evolution/"
CREATE_GIF = False

# Constants
SIM_TIME = settings["Simulation"]["SIM_TIME"]           # Simulation time in seconds
N_AGENTS = settings["Simulation"]["N_AGENTS"]           # Number of agents
N_AGENTS = 8                                            # Force number of agents to specified value
U_MAX = settings["Simulation"]["U_MAX"]                 # Maximum control saturation point
SEED = 1                                                # Set seed

# Fitness gains
k_p = settings["Simulation"]["k_p"]                     # Target position error gain
k_v = settings["Simulation"]["k_v"]                     # Target velocity error gain
k_c = settings["Simulation"]["k_c"]                     # Connectivity maintenance gain
k_u = settings["Simulation"]["k_u"]                     # Control effort gain

# Simtest
config_path = 'evolution\\networks\\neat_config_tmp'
neat_tools.dict_to_config(config_path, settings)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
net = neat.nn.FeedForwardNetwork.create(genome, config)

# Run the given simulation.
flock = Flock(2, N_AGENTS, U_MAX, SEED, SIM_TIME, k_p=k_p, k_v=k_v, k_c=k_c, k_u=k_u)
    
while flock.t < SIM_TIME and not flock.end_sim:
    flock.U = np.zeros((flock.N_AGENTS,3))
    for agent in range(N_AGENTS):
        # Get aircraft states
        inputs = flock.get_inputs(agent)
        # Apply inputs to ANN
        control = net.activate(inputs)
        # Apply control to simulation
        flock.update(agent, control)
    flock.t += flock.dt
    flock.save_data()

# Evaluate genome fitness
flock.print_report()

# Visualize
plots.animate_3d(flock, CREATE_GIF, gif_path)