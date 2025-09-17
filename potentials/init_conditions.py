# init_conditions.py
import numpy as np

def init_lennard_jones(n_agents, SEED, min_dist, max_dist):
    if SEED is not None:
        np.random.seed(SEED)
    X = generate_positions_with_spacing(n_agents, min_dist, max_dist, box_size=10.0, SEED=SEED)
    V = np.zeros((n_agents, 3))
    params = {"a": 8.0, "b": 7.0, "r0": 1.2}
    return X, V, params

def init_quadratic(n_agents, SEED, min_dist, max_dist):
    if SEED is not None:
        np.random.seed(SEED)
    X = generate_positions_with_spacing(n_agents, min_dist, max_dist, box_size=10.0, SEED=SEED)
    V = np.zeros((n_agents, 3))
    params = {"k": 1.0, "r0": 5.0}
    return X, V, params

def init_morse(n_agents, SEED, min_dist, max_dist):
    if SEED is not None:
        np.random.seed(SEED)
    X = generate_positions_with_spacing(n_agents, min_dist, max_dist, box_size=10.0, SEED=SEED)
    V = np.zeros((n_agents, 3))
    params = {"D": 2.5, "a": 1.0, "r0": 1.0}
    return X, V, params

# Map potential names to init functions
INIT_FUNCS = {
    "lennard_jones": init_lennard_jones,
    "quadratic": init_quadratic,
    "morse": init_morse,
}

def generate_positions_with_spacing(n_agents, min_dist, max_dist, box_size=10.0, SEED=None):
    """
    Generate n_agents positions in 3D such that all pairwise distances are between min_dist and max_dist.
    """
    if SEED is not None:
        np.random.seed(SEED)
    positions = []
    trials = 0
    while len(positions) < n_agents and trials < 10000:
        candidate = np.random.uniform(0, box_size, size=3)
        if all(min_dist <= np.linalg.norm(candidate - np.array(p)) <= max_dist for p in positions):
            positions.append(candidate)
        trials += 1
    if len(positions) < n_agents:
        raise RuntimeError("Could not place all agents with given spacing constraints.")
    return np.array(positions)