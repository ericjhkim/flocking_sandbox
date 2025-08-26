"""
Generic base class for agents in a multi-agent system.
"""

import numpy as np
import scipy # Required for networkx
import networkx as nx
from networkx.algorithms.tree.mst import random_spanning_tree
from networkx.algorithms import isomorphism
import itertools
from abc import ABCMeta, abstractmethod
import random

class Agents(metaclass=ABCMeta):
    def __init__(self, N_INT, SUBGRAPHS, N_AGENTS, SEED, D_MIN=5, D_MAX=25, R_MAX=30, dt=0.1, T_VEC=[0,0,0], VERBOSE=True):
        # Initialize parameters
        self.N_INT = N_INT                                  # Number of integrators
        self.SUBGRAPHS = SUBGRAPHS                          # Enable/disable subgraphs
        self.N_AGENTS = N_AGENTS                            # Number of agents
        self.SEED = SEED                                    # Seed
        if self.SEED:
            np.random.seed(self.SEED)                       # Set seed
            # T_VEC = [20,20,0]                               # Reset translation vector
        else:
            np.random.seed()                                # Random seed
        self.D_MIN = D_MIN                                  # Minimum distance between agents (for random generation)
        self.D_MAX = D_MAX                                  # Maximum distance between agents (for random generation)
        self.R_MAX = R_MAX                                  # Maximum sensing radius of agents
        self.dt = dt                                        # Time step
        self.T_VEC = T_VEC                                  # Translation vector
        self.VERBOSE = VERBOSE                              # Enable/disable print statements

        # Set dynamics based on the number of integrators
        if N_INT == 1:
            self.dynamics = self.dynamics_single
        elif N_INT == 2:
            self.dynamics = self.dynamics_double

        # Initial graph
        self.X = self.generate_3d_coordinates()             # Agent positions
        self.V = np.zeros((N_AGENTS, 3))                    # Agent velocities (zero initial)
        self.A_T1 = np.zeros((N_AGENTS, N_AGENTS))          # Target adjacency matrix (default to zeros if subgraph isomorphism is disabled)
        self.U = np.zeros((N_AGENTS, 3))

        # Subgraph isomorphisms
        if self.SUBGRAPHS:
            self.generate_subgraphs()

        # Data storage
        self.data = {
            "position": [],
            "velocity": [],
            "adjacency": [],
            "control": []
        }

    @abstractmethod
    def control(self, t):
        """
        Control law for agents.
        """
        pass

    def update(self, t):
        """
        Update agent states.
        """
        self.U = self.control(t)

        self.V = self.dynamics(self.U)
        self.X += self.V * self.dt  # Update positions
        
        self.A1 = self.get_adjacency(self.X)

        if self.VERBOSE and len(self.subgraph_connected()) > 0:
            print(f"Graph is disconnected at time {round(t,1)} for edges {self.subgraph_connected()}.")

        # Save data
        self.save_data()

    def dynamics_single(self, U):
        """
        Single-integrator dynamics.
        """
        return U
    
    def dynamics_double(self, U):
        """
        Double-integrator dynamics.
        """
        return self.V + U * self.dt
    
    def generate_subgraphs(self):
        self.A1 = self.get_adjacency(self.X)                # Adjacency matrix
        self.G1 = nx.from_numpy_array(self.A1)              # NetworkX graph
        # Target graph
        self.X2 = self.generate_3d_coordinates(translation=self.T_VEC)
        self.A2 = self.get_adjacency(self.X2)               # Adjacency matrix
        self.G2 = nx.from_numpy_array(self.A2)              # NetworkX graph

        # Check match
        found = False
        for a in range(50):                                 # Try 50 times to find a matching tree
            T1 = random_spanning_tree(self.G1, seed=self.SEED)
            match, T2 = self.check_tree_dynamically(T1, self.G2)
            if match:
                found = True
                break
            elif self.VERBOSE:
                print(f"Failed to find a matching tree in attempt {a}.")
                continue
        
        if found:
            matcher = nx.isomorphism.GraphMatcher(T1, T2)
            matches = matcher.is_isomorphic()
            self.mapping = matcher.mapping
            if self.VERBOSE:
                print("Node Mapping from T1 to T2: ", self.mapping, " | Is isomorphic: ", matches)

            self.A_T1 = nx.adjacency_matrix(T1, nodelist=sorted(T1.nodes()), dtype=np.float64).toarray()
            self.A_T2 = nx.adjacency_matrix(T2, nodelist=sorted(T2.nodes()), dtype=np.float64).toarray()
        elif self.VERBOSE:
            print("No matching tree found")

        # Create an output array with the same shape
        self.X_tgt = np.zeros_like(self.X2)

        # Rearrange rows based on mapping
        for old_idx, new_idx in self.mapping.items():
            self.X_tgt[old_idx] = self.X2[new_idx]

    def subgraph_connected(self):
        """
        Check if a subgraph is connected.
        """
        M = self.A1 - self.A_T1
        neg = np.where(M < 0)
        neg = [(int(row), int(col)) for row, col in zip(neg[0], neg[1])]
        return neg

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
        trials = 0
        while len(coordinates) < self.N_AGENTS and trials < 1e3:
            new_point = np.random.uniform(0, self.D_MAX, size=3) + translation
            if is_valid_point(new_point, np.array(coordinates)):
                coordinates.append(new_point)
            trials += 1

        return np.array(coordinates)

    def get_adjacency(self, X):
        """
        Compute the adjacency matrix for a given set of agent positions.
        """
        A = np.zeros((self.N_AGENTS,self.N_AGENTS))
        for i in range(self.N_AGENTS):
            for j in range(self.N_AGENTS):
                if np.linalg.norm(X[i,:3] - X[j,:3]) <= self.R_MAX:
                    A[i,j] = 1
                if i == j:
                    A[i,j] = 0
        return A
    
    def save_data(self):
        """
        Store state in data storage object
        """
        self.data["position"].append(np.array(self.X))
        self.data["velocity"].append(np.array(self.V))
        self.data["adjacency"].append(self.A1)
        self.data["control"].append(self.U)