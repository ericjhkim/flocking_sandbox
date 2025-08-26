# init_conditions.py
import numpy as np

def init_lennard_jones(n_agents):
    X = np.random.rand(n_agents, 3)
    V = np.zeros((n_agents, 3))
    params = {"a": 0.2, "b": 2.5, "r0": -0.15}
    return X, V, params

def init_quadratic(n_agents):
    X = np.random.rand(n_agents, 3)
    V = np.zeros((n_agents, 3))
    params = {"k": 1.0, "r0": 5.0}
    return X, V, params

def init_morse(n_agents):
    X = np.random.rand(n_agents, 3)
    V = np.zeros((n_agents, 3))
    params = {"D": 2.5, "a": 1.0, "r0": 1.0}
    return X, V, params

# Map potential names to init functions
INIT_FUNCS = {
    "lennard_jones": init_lennard_jones,
    "quadratic": init_quadratic,
    "morse": init_morse,
}
