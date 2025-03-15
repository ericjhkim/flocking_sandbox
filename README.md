# Flocking Sandbox

This is a simulation sandbox for works-in-progress.

## Showcase
### Laplacian-Pinning Connectivity Maintenance ([laplacian_pinning.py](/laplacian_pinning.py))
Building off [1], the goal is to preserve agent connectivity between initial and final swarm configurations (both relative and global reference frame positions).
The control is implemented in a single-integrator system:

$$\dot{X} = -k_1 L (X-X_{tgt}) - k_2 P (X-X_{tgt})$$

Where $X$ is the $N\times3$ matrix of agents' positions in 3D Euclidean space, $X_{tgt}$ is the $N\times3$ matrix of the agents' final target positions, $L$ is the system's graph Laplacian matrix, $P$ is a diagonal matrix of pins, where $P_{ii} = 1$ for pinning agents and 0 otherwise, and $k_1$ and $k_2$ are gains. The first term accounts for convergence towards the relative positioning of agents in the swarm configuration, and the second term accounts for convergence towards the global frame position.

![sim1](https://github.com/ericjhkim/flocking_sandbox/blob/main/visualizations/laplacian_pinning/anim_20250315_152244.gif)

### Laplacian-Pinning with Spring System Connectivity Maintenance ([laplacian_pinning_spring.py](/laplacian_pinning_spring.py))
An extension of the above control, this variant of the Laplacian-Pinning adds a spring-dyanmics potential function to experiment with more flexible connectivity behaviour.
The control is implemented in a double-integrator system:

$$\ddot{X_1} = -k_1 L (X-X_{tgt}) - k_2 P (X-X_{tgt})$$

$$\ddot{X_2} = \sum_{j\in N_i}K_{ij}(||x_j-x_i||-L_{ij})\frac{x_j-x_i}{||x_j-x_i||} - k_3 \dot{x}_i$$

$$\ddot{X} = \ddot{X_1} + \ddot{X_2}$$

Where $K$ is a matrix of spring constants, defined as the adjacency matrix for the randomly-generated isomorphic subgraph $T_1$, and $L$ is the natural spring length, set to be the relative distance between two agents in the final target formation $X_{tgt}$.

![sim2](https://github.com/ericjhkim/flocking_sandbox/blob/main/visualizations/laplacian_pinning_spring/anim_20250315_154650.gif)

## References
  1. Hamaoui, M. (2024). *Connectivity Maintenance through Unlabeled Spanning Tree Matching*. J Intell Robot Syst 110, 15 [doi:10.1007/s10846-024-02048-9](https://doi.org/10.1007/s10846-024-02048-9)(X-X_{tgt})