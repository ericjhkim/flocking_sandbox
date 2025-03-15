"""
This is double-integrator dynamics for a Laplacian-Pinning control law (control stable) augmented by a mass-spring system.

@author: Kim.ej
"""

# 3D Simulation Setup
import os
import numpy as np
import scipy            # Required for networkx!
from tqdm import tqdm
from PIL import Image
import tools as tools
from datetime import datetime
import networkx as nx
from networkx.algorithms.tree.mst import random_spanning_tree
from networkx.algorithms import isomorphism
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Directories
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
gif_path = f"visualizations/anim_{TIMESTAMP}.gif"

# Controls

SAVE_DATA = False
CREATE_GIF = False
SEED = 1
# np.random.seed(SEED)

# Constants
dt = 0.1                                    # Simulation interval
SIM_TIME = 20                               # Simulation time in seconds
NUM_STEPS = int(SIM_TIME / dt)              # Number of simulation steps

N_AGENTS = 8                                # Number of agents
R_MAX = 25                                  # Distance at which agents can sense each other
D_MIN = 5                                   # Minimum distance between agents (for random generation of initial position)
D_MAX = R_MAX*1.5                           # Maximum distance between agents (for random generation of initial position)
MAX_VEL = 10.0                              # Maximum velocity of agents

k1 = 1.0                                    # Relative formation control gain
k2 = 1.0                                    # Absolute position control gain

def main():
    # Set pin(s)
    pins = np.arange(0, N_AGENTS)

    # Create an instance of the class
    flock = Flocking3D(N_AGENTS, pins, [k1, k2], R_MAX, D_MIN, D_MAX, MAX_VEL, dt)

    # Run animation for 3D
    for t in np.arange(0, SIM_TIME, dt):
        flock.update(t)

    tools.animate_3d(flock)

class Flocking3D:
    def __init__(self, N_AGENTS, pins, k, R_MAX, D_MIN, D_MAX, MAX_VEL, dt):
        # Initialize parameters
        self.N_AGENTS = N_AGENTS
        self.k1, self.k2 = k
        self.R_MAX = R_MAX
        self.D_MIN = D_MIN                                  # Minimum distance between agents (for random generation)
        self.D_MAX = D_MAX                                  # Maximum distance between agents (for random generation)
        self.MAX_VEL = MAX_VEL
        self.dt = dt

        # Pins and global position
        self.P = np.diag(np.zeros(self.N_AGENTS))           # Pinning matrix
        self.P[pins,pins] = 1

        self.T_VEC = [20.0, 20.0, 0.0]                      # Translation vector (for random generation of initial position)

        # Initial graph
        self.X = self.generate_3d_coordinates()
        self.V = np.zeros((N_AGENTS, 3))                    # Agent velocities (zero initial)
        self.A1 = self.get_adjacency(self.X)
        self.G1 = nx.from_numpy_array(self.A1)
        # tools.draw_graph(self.G1)
        # tools.plot_agents_3d(self.X,self.A1)

        # Target graph
        self.X2 = self.generate_3d_coordinates(translation=self.T_VEC)
        self.A2 = self.get_adjacency(self.X2)
        self.G2 = nx.from_numpy_array(self.A2)
        # tools.plot_agents_3d(self.X2,self.A2)

        # Check match
        found = False
        for a in range(50):
            # try:
            T1 = random_spanning_tree(self.G1, seed=SEED)
            match, T2 = self.check_tree_dynamically(T1, self.G2)
            if match:
                found = True
                break
            else:
                print(f"Failed to find a matching tree in attempt {a}.")
                continue
            # except:
            #     continue
        
        if found:
            print("Found a matching tree.")
            matcher = nx.isomorphism.GraphMatcher(T1, T2)
            matches = matcher.is_isomorphic() # Run this before the mapping
            self.mapping = matcher.mapping
            print("Node Mapping from T1 to T2: ", self.mapping, " | Is isomorphic: ", matches)

            self.A_T1 = nx.adjacency_matrix(T1, nodelist=sorted(T1.nodes()), dtype=np.float64).toarray()
            # print(self.A_T1)
            # print(f"G1's Adjacency Matrix:\n{self.A1}")
            # print(f"G1 Subgraph's Adjacency Matrix:\n{self.A_T1}")

            self.A_T2 = nx.adjacency_matrix(T2, nodelist=sorted(T2.nodes()), dtype=np.float64).toarray()
            # print(self.A_T2)
            # print(f"G2's Adjacency Matrix:\n{self.A2}")
            # print(f"G2 Subgraph's Adjacency Matrix:\n{self.A_T2}")

            # tools.draw_graphs([self.G1, T1, self.G2, T2])
            # tools.plot_agents_connectivity(self.X, self.A1, self.A_T1)
        else:
            print("No matching tree found")

        # Create an output array with the same shape
        self.X_tgt = np.zeros_like(self.X2)

        # Rearrange rows based on mapping
        for old_idx, new_idx in self.mapping.items():
            self.X_tgt[old_idx] = self.X2[new_idx]

        self.X_global = np.zeros((N_AGENTS, 3))
        self.X_global[pins] = self.X_tgt[pins]

        ## Spring stuff
        self.L1 = np.zeros((self.N_AGENTS, self.N_AGENTS))      # Natural spring length matrix
        for i in range(self.N_AGENTS):
            for j in range(self.N_AGENTS):
                self.L1[i, j] = np.linalg.norm(self.X_tgt[i] - self.X_tgt[j])
        self.K1 = self.A1 + self.A_T1                           # Spring constant matrix

        # Data storage
        self.data = [np.array(self.X)]

    def update(self, t):
        U = np.zeros((self.N_AGENTS, 3))
        for i in range(self.N_AGENTS):
            for j in range(self.N_AGENTS):
                if self.A1[i, j] != 0:  # If neighbours
                    U[i] += self.K1[i,j]*(np.linalg.norm(self.X[j] - self.X[i]) - self.L1[i,j]) * (self.X[j] - self.X[i])/np.linalg.norm(self.X[j] - self.X[i])
            U[i] -= 2*self.V[i]

        # Compute laplacian
        L = self.compute_laplacian(self.get_adjacency(self.X))

        # Compute formation control
        error = self.X - self.X_tgt  # Error in position
        U += -self.k1 * L @ error - self.k2 * self.P @ error

        self.V += U * self.dt
        self.X += self.V * self.dt  # Update positions

        if not self.fiedler_check(self.X):
            print(f"Graph is disconnected at time {t}.")

        # Save data
        self.save_data()

    def compute_laplacian(self, A):
        """
        Compute the Laplacian matrix from a given adjacency matrix.
        """
        D = np.diag(A.sum(axis=1))  # Compute the degree matrix
        L = D - A  # Laplacian matrix
        return L
    
    def fiedler_check(self, X):
        adj_matrix = self.get_adjacency(X)
        L = np.diag(adj_matrix.sum(axis=1)) - adj_matrix  # Laplacian
        eigenvalues = np.linalg.eigvalsh(L)  # Compute eigenvalues
        lambda2 = np.sort(eigenvalues)[1]  # Second smallest eigenvalue
        if lambda2 > 0:
            return True
        else:
            return False
    
    def check_tree_dynamically(self, T1, G2):
        """
        Check if a spanning tree T1 (of G1) is isomorphic to any spanning tree of another graph (G2)
        without precomputing all spanning trees of G2.
        """
        edges = list(G2.edges())
        for edge_subset in itertools.combinations(edges, len(G2.nodes) - 1):
            candidate_tree = G2.edge_subgraph(edge_subset).copy()
            if nx.is_tree(candidate_tree):
                if self.is_tree_isomorphic(T1, candidate_tree):
                    return True, candidate_tree
        return False, None
    
    def is_tree_isomorphic(self, tree1, tree2):
        """Check if two trees are isomorphic."""
        gm = isomorphism.GraphMatcher(tree1, tree2)
        return gm.is_isomorphic()

    def generate_3d_coordinates(self, translation=[0,0,0]):
        """
        Generate N 3D coordinates with a minimum distance of D_MIN and a maximum distance of D_MAX.
        This is to nondeterministically initialize agents' locations.
        """
        def is_valid_point(new_point, points):
            if len(points) == 0:
                return True
            distances = np.linalg.norm(points - new_point, axis=1)
            return np.all((distances >= self.D_MIN) & (distances <= self.D_MAX))

        coordinates = []

        # Create points iteratively
        while len(coordinates) < self.N_AGENTS:
            new_point = np.random.uniform(0, self.D_MAX, size=3)
            if is_valid_point(new_point, np.array(coordinates)):
                new_point += translation
                coordinates.append(new_point)

        return np.array(coordinates)

    def get_adjacency(self, states):
        """
        Compute the adjacency matrix for a given set of agent states.
        """
        A = np.zeros((self.N_AGENTS,self.N_AGENTS))
        for i in range(self.N_AGENTS):
            for j in range(self.N_AGENTS):
                if np.linalg.norm(states[i,:3] - states[j,:3]) <= self.R_MAX:
                    A[i,j] = 1
                if i == j:
                    A[i,j] = 0
        return A
    
    def save_data(self):
        """
        Store state in data storage object
        """
        self.data.append(np.array(self.X))

if __name__ == "__main__":
    main()