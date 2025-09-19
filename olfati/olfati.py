"""
Implementation of Olfati-Saber's potential function algorithm from "Flocking for Multi-Agent Dynamic Systems"
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tools import plots, agents

# Controls
gif_path = "visualizations/"
CREATE_GIF = True
SEED = 1

# Constants
SIM_TIME = 5                                # Simulation time in seconds
N_AGENTS = 8                                # Number of agents

c_alpha_1 = 1                               # Position-based attraction/repulsion between agents
c_beta_1 = 3                                # Position-based attraction/repulsion for obstacles
c_gamma_1 = 1                               # Navigational feedback term (position)
k_a = 0.9                                   # Potential depth (attraction strength) 0 < a <= b
k_b = 1                                     # Repulsion strength

def main():
    # Create flock
    flock = Flock(2, N_AGENTS, SEED)

    # Run simulation
    for t in np.arange(0, SIM_TIME, flock.dt):
        flock.update(t)

    plots.animate_3d(flock, CREATE_GIF, gif_path, follow=True, follow_padding=3.0, cartoon=True)

class Flock(agents.Agents):
    def __init__(self, N_INT, N_AGENTS, SEED):
        super().__init__(N_INT, False, N_AGENTS, SEED, T_VEC=[20,20,0])
        self.alg = Algorithms(self.R_MAX,c_alpha_1,c_beta_1,c_gamma_1,k_a,k_b)

        self.V = np.random.uniform(-10,10,size=(N_AGENTS,3))
        self.X_tgt = self.X.copy() + np.array([1e3,1e3,0.0])
        self.V_tgt = np.array([2.0,1.0,0.0])

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
                    p_j = self.V[j]
                    U[i] += self.alg.u_alpha(q_j,q_i,p_j,p_i)
            U[i] += self.alg.u_gamma(q_i,self.X_tgt[i],p_i,self.V_tgt)

        return U

class Algorithms:
    def __init__(self,R_MAX,c_alpha_1=1,c_beta_1=3,c_gamma_1=4,a=3,b=4):
        r = R_MAX
        self.r = r                                      # Interaction range for agents
        self.r_prime = r                                # Interaction range for obstacle
        self.d = 0.5*r                                  # Desired distance between agents
        self.d_prime = 0.75*r                           # Desired distance between agent and obstacle

        self.c_alpha_1 = c_alpha_1                      # Position-based attraction/repulsion between agents
        self.c_alpha_2 = 2*np.sqrt(self.c_alpha_1)      # Velocity-based alignment between agents
        self.c_beta_1 = c_beta_1                        # Position-based attraction/repulsion for obstacles
        self.c_beta_2 = 2*np.sqrt(self.c_beta_1)        # Velocity-based alignment for obstacles
        self.c_gamma_1 = c_gamma_1                      # Navigational feedback term (position)
        self.c_gamma_2 = 1                              # Navigational feedback term (velocity)

        self.a = a                                      # Potential depth (attraction strength) 0 < a <= b
        self.b = b                                      # Repulsion strength
        self.c = np.abs(self.a-self.b)/np.sqrt(4*self.a*self.b)

        self.h = 0.2                                    # Bump function parameter - interaction cutoff (smaller = sharper cutoff, larger = more fluid)
        self.epsilon = 0.1

    def u_alpha(self,q_j,q_i,p_j,p_i):
        u_i = self.c_alpha_1*self.phi_alpha(self.sigma_norm(q_j-q_i))*self.n_ij(q_j,q_i) + self.c_alpha_2*self.a_ij(q_j,q_i)*(p_j-p_i)
        return u_i

    def u_beta(self,q_k,q_i,p_k,p_i):
        u_i = self.c_beta_1*self.phi_beta(self.sigma_norm(q_k-q_i))*self.n_ij(q_k,q_i) + self.c_beta_2*self.b_ik(q_k,q_i)*(p_k-p_i)
        return u_i
    
    # Navigational feedback term
    def u_gamma(self,q_i,q_r,p_i,p_r):
        u_i = -self.c_gamma_1*self.sigma_1(q_i-q_r) - self.c_gamma_2*(p_i-p_r)
        return u_i

    def sigma_1(self,z):
            return z/np.sqrt(1+z**2)

    # Action function (eq. 15)
    def phi_alpha(self,z):
        def phi(z):
            return 0.5*((self.a+self.b)*self.sigma_1(z+self.c)+(self.a-self.b))
        
        r_alpha = self.sigma_norm(self.r)
        d_alpha = self.sigma_norm(self.d)
        
        return self.rho_h(z/r_alpha) * phi(z-d_alpha)

    def phi_beta(self,z):
        d_beta = self.sigma_norm(self.d_prime)
        
        return self.rho_h(z/d_beta) * (self.sigma_1(z-d_beta)-1)

    # Bump function (eq. 10)
    def rho_h(self,z):
        if z >= 0 and z < self.h:
            return 1
        elif z >= self.h and z <= 1:
            return 0.5*(1+np.cos(np.pi*(z-self.h)/(1-self.h)))
        else:
            return 0
        
    # Sigma norm (eq. 8)
    def sigma_norm(self,z):
        return (np.sqrt(1+self.epsilon*np.linalg.norm(z)**2)-1)/self.epsilon

    # Gradient of the sigma norm (sigma_epsilon(z): eq. 9)
    def grad_sigma_norm(self,z):
        return z/np.sqrt(1+self.epsilon*np.linalg.norm(z)**2)

    def n_ij(self,q_j,q_i):
        return self.grad_sigma_norm(q_j-q_i)
    
    # Spatial adjacency matrix A(q)
    def a_ij(self,q_j,q_i):
        r_alpha = self.sigma_norm(self.r)
        return self.rho_h(self.sigma_norm(q_j-q_i)/r_alpha)

    def b_ik(self,q_k,q_i):
        d_beta = self.sigma_norm(self.d_prime)
        return self.rho_h(self.sigma_norm(q_k-q_i)/d_beta)

if __name__ == "__main__":
    main()