"""
This script simulates the flocking behavior of agents in 3D space using the holonomic model.
The control law is based on relative formation and global position.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tools import plots, agents
from scipy.linalg import expm, logm

# Controls
gif_path = "visualizations/geometric_relative/"
CREATE_GIF = False
SEED = 1

# Constants
SIM_TIME = 10                               # Simulation time in seconds
N_AGENTS = 8                                # Number of agents

k_r = 1.0                                   # Relative position tracking gain
k_p = 1.0                                   # Pinning gain
k_d = 2.0                                   # Velocity damping gain

def main():
	# Set pin(s)
	pins = np.arange(0, N_AGENTS)
	pins = [0,2,7]

	# Create an instance of the class
	flock = Flock(2, N_AGENTS, [k_r, k_p, k_d], pins, SEED)

	# Run animation for 3D
	for t in np.arange(0, SIM_TIME, flock.dt):
		flock.update(t)

	plots.animate_3d(flock, CREATE_GIF, gif_path)

class Flock(agents.Agents):
	def __init__(self, N_INT, N_AGENTS, gains, pins, SEED):
		super().__init__(N_INT, True, N_AGENTS, SEED, T_VEC=[20,20,0])

		# These functions are necessary, because the state X (and V) contains [3d pose, 3d position]
		self.X = self.generate_3d_pose()
		self.V = np.zeros((self.N_AGENTS, 6))
		self.X2 = self.generate_3d_pose(translation=self.T_VEC)

		# Control gains
		self.k_r, self.k_p, self.k_d = gains

		# Pins and global position
		self.pins = pins
		self.P = np.diag(np.zeros(self.N_AGENTS))           # Pinning matrix
		self.P[pins,pins] = 1

		# Create an output array with the same shape
		self.X_goal = np.zeros_like(self.X2)

		# Rearrange rows based on mapping
		for old_idx, new_idx in self.mapping.items():
			self.X_goal[old_idx] = self.X2[new_idx]

		self.X_tgt = self.X_goal[:, 3:]
		
		# Data storage
		self.data = {
			"pose": [np.array(self.X)[:,:3]],
			"position": [np.array(self.X)[:,3:]],
			"velocity": [np.array(self.V)[:,3:]],
			"adjacency": [self.get_adjacency(self.X[:,3:])],
			"control": [self.U]
		}

	def update(self, t):
		# Compute laplacian
		self.A1 = self.get_adjacency(self.X[:, 3:])

		# Update dynamics
		for i in range(self.N_AGENTS): 

			self.U = self.control(i)
			
			self.V[i] += self.U * self.dt
			v_hat = hat_map_SE3(self.V[i])
			self.X[i] = out_SE3(in_SE3(self.X[i]) @ exp_map_SE3(v_hat * self.dt))

		if len(self.subgraph_connected()) > 0:
			print(f"Graph is disconnected at time {round(t,1)} for edges {self.subgraph_connected()}.")

		# Save data
		self.save_data()
	
	def control(self, i):
		"""
		This is a Laplacian-based control law for a pinned holonomic system.
		"""
		u = np.zeros(6)
		
        # Relative term (formation control)
		for j in range(self.N_AGENTS):
			if j != i and self.A1[i, j] == 1:
				gij = np.linalg.inv(in_SE3(self.X[i])) @ in_SE3(self.X[j])
				gij_des = np.linalg.inv(in_SE3(self.X_goal[i])) @ in_SE3(self.X_goal[j])
				gij_des_inv = np.linalg.inv(gij_des)  # desired relative pose
				error = log_map_SE3(gij @ gij_des_inv)
				u += k_r * vee_map_SE3(error)

        # Pinning term (global position)
		if self.P[i, i] == 1:
			gi0 = np.linalg.inv(in_SE3(self.X[i])) @ in_SE3(self.X_goal[i])
			error_i0 = log_map_SE3(gi0)
			u += k_p * vee_map_SE3(error_i0)

        # Velocity damping
		u -= k_d * self.V[i]

		return u
	
	def subgraph_connected(self):
		"""
		Check if a subgraph is connected.
		"""
		A = self.get_adjacency(self.X[:, 3:])
		M = A - self.A_T1
		neg = np.where(M < 0)
		neg = [(int(row), int(col)) for row, col in zip(neg[0], neg[1])]
		return neg
	
	def fiedler_check(self, X):
		adj_matrix = self.get_adjacency(X[:, 3:])
		L = np.diag(adj_matrix.sum(axis=1)) - adj_matrix  # Laplacian
		eigenvalues = np.linalg.eigvalsh(L)  # Compute eigenvalues
		lambda2 = np.sort(eigenvalues)[1]  # Second smallest eigenvalue
		if lambda2 > 0:
			return True
		else:
			return False

	def generate_3d_pose(self, translation=[0,0,0]):
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
			new_point = np.random.uniform(0, self.D_MAX, size=3) + translation
			if is_valid_point(new_point, np.array(coordinates)):
				coordinates.append(new_point)

		attitudes = np.random.uniform(0, 2*np.pi, size=(self.N_AGENTS,3))
		pose = np.hstack((attitudes, coordinates))
		return np.array(pose)
	
	def save_data(self):
		"""
		Store state in data storage object
		"""
		self.data["pose"].append(np.array(self.X)[:,:3])
		self.data["position"].append(np.array(self.X)[:,3:])
		self.data["velocity"].append(np.array(self.V)[:,3:])
		self.data["adjacency"].append(self.A1)
		self.data["control"].append(self.U)

def in_SE3(pose):
	"""
	Converts a pose [theta_x, theta_y, theta_z, x, y, z] to an SE(3) matrix.
	"""
	theta_x, theta_y, theta_z, x, y, z = pose
	R_x = np.array([[1, 0, 0],
					[0, np.cos(theta_x), -np.sin(theta_x)],
					[0, np.sin(theta_x), np.cos(theta_x)]])
	R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
					[0, 1, 0],
					[-np.sin(theta_y), 0, np.cos(theta_y)]])
	R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
					[np.sin(theta_z), np.cos(theta_z), 0],
					[0, 0, 1]])
	R = R_z @ R_y @ R_x
	g = np.eye(4)
	g[:3, :3] = R
	g[:3, 3] = [x, y, z]
	return g

def out_SE3(g):
	"""
	Converts an SE(3) matrix to a pose [theta_x, theta_y, theta_z, x, y, z].
	"""
	R = g[:3, :3]
	x, y, z = g[:3, 3]
	theta_x = np.arctan2(R[2, 1], R[2, 2])
	theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
	theta_z = np.arctan2(R[1, 0], R[0, 0])
	return [theta_x, theta_y, theta_z, x, y, z]

def hat_map_SE3(xi):
	"""
	Converts a velocity vector [omega_x, omega_y, omega_z, v_x, v_y, v_z] to an se(3) matrix.
	"""
	omega_x, omega_y, omega_z, v_x, v_y, v_z = xi
	xi_hat = np.array([[0, -omega_z, omega_y, v_x],
					   [omega_z, 0, -omega_x, v_y],
					   [-omega_y, omega_x, 0, v_z],
					   [0, 0, 0, 0]])
	return xi_hat

def vee_map_SE3(xi_hat):
	"""
	Converts an se(3) matrix to a velocity vector [omega_x, omega_y, omega_z, v_x, v_y, v_z].
	"""
	omega_x = xi_hat[2, 1]
	omega_y = xi_hat[0, 2]
	omega_z = xi_hat[1, 0]
	v_x = xi_hat[0, 3]
	v_y = xi_hat[1, 3]
	v_z = xi_hat[2, 3]
	return np.array([omega_x, omega_y, omega_z, v_x, v_y, v_z])

def exp_map_SE3(xi_hat):
	"""
	Computes the matrix exponential of an se(3) matrix.
	"""
	return expm(xi_hat)

def log_map_SE3(g):
	"""
	Computes the matrix logarithm of an SE(3) matrix.
	"""
	return logm(g)

if __name__ == "__main__":
	main()