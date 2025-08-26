"""
Evolve ANN using NEAT algorithm.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import neat
import numpy as np
from tools import plots, agents
from time import process_time, perf_counter
import tools.neat_tools as neat_tools
from datetime import datetime

# Save parameters
net_id = datetime.now().strftime("%Y%m%d_")
files = [x for x in os.listdir("evolution/networks/") if "." not in x and x != "neat_config_tmp"]
try: last_id = int(sorted(files)[-1][-1])
except: last_id = -1
net_id += str(last_id+1)
print(f"Evolving network {net_id}")

network_description = """
Tuning fitness gains for fastest convergence to positions.
"""

test = 0
gif_path = "visualizations/neat/"
CREATE_GIF = False

# Simulation Parameters
runs_per_net = 3                            # Evaluate each genome n times
generations = 100                           # Number of evolutions

# Constants
SIM_TIME = 20                               # Simulation time in seconds
N_AGENTS = 5                                # Number of agents
U_MAX = 6                                   # Maximum control saturation point
SEED = None                                 # Random seed

# Fitness gains
k_p = 1.0                                   # Target position error gain
k_v = 1.0                                   # Target velocity error gain
k_c = 1.0                                   # Connectivity maintenance gain
k_u = 0.1                                   # Control effort gain

def main():
    t_start_real = perf_counter()

    # Load the config file
    config_path = 'evolution\\neat_config'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    # Evaluate for n generations
    t_start = process_time()                        # Mark start of evolution
    pe = neat.ParallelEvaluator(4, eval_genome)
    winner = pop.run(pe.evaluate,generations)
    t_stop = process_time()                         # Mark end of evolution

    # Preprocess pickle data (with metadata) before saving
    settings = neat_tools.config_to_dict(runs_per_net=runs_per_net, generations=generations, SIM_TIME=SIM_TIME, N_AGENTS=N_AGENTS, U_MAX=U_MAX, SEED=SEED, k_p=k_p, k_v=k_v, k_c=k_c, k_u=k_u)
    pickle_data = {'description': network_description.strip("\n"),
                   'genome': winner,
                   'settings': settings}

    # Save the winner.
    with open(f'D:\\EK Research\\flocking_sandbox\\evolution\\networks\\{net_id}', 'wb') as f:
        pickle.dump(pickle_data, f)
    print(winner)

    # Draw network
    node_names = {-1: 'X', -2: 'Y', -3: 'Z', -4: 'vX', -5: 'vY', -6: 'vZ', 0: 'uX', 1: 'uY', 2: 'uZ'}
    neat_tools.draw_net(config, winner, view=False, node_names=node_names,
                       filename=f"D:\\EK Research\\flocking_sandbox\\evolution\\networks\\{net_id}.enabled-pruned.gv", show_disabled=True, prune_unused=False)

    # Stopwatches
    cputime = round(t_stop-t_start,2)
    print('CPU time: '+str(cputime)+'s = '+str(round(cputime/60,2))+'m.') # Print CPU time
    t_stop_real = perf_counter()
    realtime = round(t_stop_real-t_start_real,2)
    print('Elapsed time: '+str(realtime)+'s = '+str(round(realtime/60,2))+'m.') # Print real world time

    # Run the winning genome in simulation.
    net = neat.nn.FeedForwardNetwork.create(winner, config)
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

def eval_genome(genome, config):
    """
    Evaluate a single genome.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for _ in range(runs_per_net):
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
        fitness = flock.get_fitness()
        fitnesses.append(fitness)
    
    # The genome's fitness is its total fitness across all runs
    final_score = np.nansum(fitnesses)
    return final_score
    
def eval_genomes(genomes, config):
    """
    Evaluate a list of genomes.
    """
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
    
class Flock(agents.Agents):
    # List permitted kwargs
    __allowed = ("k_p", "k_v", "k_c", "k_u", "k_t")
    def __init__(self, N_INT, N_AGENTS, U_MAX, SEED, SIM_TIME, VERBOSE=False, **kwargs):
        super().__init__(N_INT, True, N_AGENTS, SEED, T_VEC=np.random.uniform(0,50,3), VERBOSE=VERBOSE)
        # Time
        self.SIM_TIME = SIM_TIME
        self.t = 0.0
        self.end_sim = 0
        self.U_MAX = U_MAX

        # Fitness gains
        for k, v in kwargs.items():
            assert(k in self.__class__.__allowed)
            setattr(self, k, v)

        self.conns = [self.calculate_connectivity()]

    def control(self):
        pass

    def update(self, agent, U):
        """
        Update agent states.
        """
        # Rescale control
        self.U[agent] = np.array(U)*self.U_MAX - self.U_MAX/2

        self.V[agent] += self.U[agent] * self.dt
        self.X[agent] += self.V[agent] * self.dt # Update positions

        self.continue_sim()

    def save_data(self):
        """
        Extend data storage object
        """
        super().save_data()
        self.conns.append(self.calculate_connectivity())

    def continue_sim(self):
        """
        Check if the goal has been reached or unforgiveable penalties have occurred.
        """
        if len(self.data["position"]) < 1:
            return

        position_error = np.mean( (np.linalg.norm(np.array(self.data["position"][-1]) - np.array(self.X_tgt), axis=1) / np.linalg.norm(np.array(self.data["position"][0]) - np.array(self.X_tgt), axis=1))**2 )
        velocity_error = np.mean( np.linalg.norm(self.data["velocity"][-1], axis=1)**2 )

        # If flock position and velocity error within tolerance
        if ((position_error**2) + (velocity_error**2)) < 1e-3:
            self.end_sim = 1

        # If graph disconnected
        if self.conns[-1] != 0.0:
            self.end_sim = 2

    def get_inputs(self, agent):
        """
        Fetch inputs for the neural network.
        """
        inputs = np.concatenate((self.X_tgt[agent]-self.X[agent],self.V[agent]))

        return inputs
    
    def calculate_connectivity(self):
        """
        Calculate the connectivity of the flock at the current timestep.
        """
        connectivity = []
        for i in range(self.N_AGENTS):
            for j in range(self.N_AGENTS):
                if i != j:
                    if self.A_T1[i,j] == 1 and self.A1[i,j] != 1:       # Add penalty if disconnected when should be connected
                        connectivity.append(1.0)
                    elif self.A_T1[i,j] == 1:                           # No penalty if connected when should be connected
                        connectivity.append(0.0)
        return np.mean(connectivity)
    
    def get_fitness(self):
        """
        Calculate fitness of the simulation.
        """
        position_error = np.mean( (np.linalg.norm(np.array(self.data["position"][-1]) - np.array(self.X_tgt), axis=1) / np.linalg.norm(np.array(self.data["position"][0]) - np.array(self.X_tgt), axis=1))**2 )
        velocity_error = np.mean( np.linalg.norm(self.data["velocity"][-1], axis=1)**2 )

        # Connectivity error
        if self.conns[-1] != 0.0:
            connectivity_error = 1e7**2                                 # Extreme penalty for disconnected graph
        else:
            connectivity_error = 0

        # Control effort
        control_effort = np.mean( (np.linalg.norm(self.data["control"], axis=2) / np.linalg.norm(self.U_MAX*np.ones(3)))**2, )
        # Bonus for reaching the goal before the end of the simulation
        # time_bonus = self.SIM_TIME
        # if self.end_sim:
        #     time_bonus = self.t

        return -( self.k_p * position_error +
                  self.k_v * velocity_error +
                  self.k_c * connectivity_error +
                  self.k_u * control_effort )
    
    def print_report(self):
        """
        Print simulation report.
        """
        fitness = self.get_fitness()
        if self.end_sim == 1:
            print(f"Simulation ended prematurely (goal reached) at {np.round(self.t,1)} seconds with fitness {np.round(fitness,2)}.")
        elif self.end_sim == 2:
            print(f"Simulation ended prematurely (graph disconnected) at {np.round(self.t,1)} seconds with fitness {np.round(fitness,2)}.")
        else:
            print(f"Simulation ended at {np.round(self.t,1)} seconds with fitness {np.round(fitness,2)}.")

    
if __name__ == '__main__':
    main()
