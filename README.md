# Flocking Sandbox

This is a simulation sandbox for works-in-progress.

## Showcase
### Laplacian-Pinning Connectivity Maintenance ([laplacian_pinning.py](/laplacian/laplacian_pinning.py))
Building off [1], the goal is to preserve agent connectivity between initial and final swarm configurations (both relative and global reference frame positions).
The control is implemented in a single-integrator system:

$$U = \dot{X} = -k_1 L (X-X_{tgt}) - k_2 P (X-X_{tgt})$$

Where $X$ is the $N\times3$ matrix of agents' positions in 3D Euclidean space, $X_{tgt}$ is the $N\times3$ matrix of the agents' final target positions, $L$ is the system's graph Laplacian matrix, $P$ is a diagonal matrix of pins, where $P_{ii} = 1$ for pinning agents and 0 otherwise, and $k_1$ and $k_2$ are gains. The first term accounts for convergence towards the relative positioning of agents in the swarm configuration, and the second term accounts for convergence towards the global frame position.

<img src="https://github.com/ericjhkim/flocking_sandbox/blob/main/visualizations/laplacian_pinning/anim_20250315_152244.gif" style="width:75%;">

### Laplacian-Pinning with Spring System Connectivity Maintenance ([laplacian_pinning_spring.py](/laplacian/laplacian_pinning_spring.py))
An extension of the above control, this variant of the Laplacian-Pinning adds a spring-dyanmics potential function to experiment with more flexible connectivity behaviour.
The control is implemented in a double-integrator system:

$$\ddot{X_1} = -k_1 L (X-X_{tgt}) - k_2 P (X-X_{tgt})$$

$$\ddot{X_2} = \sum_{j\in N_i}K_{ij}(||x_j-x_i||-L_{ij})\frac{x_j-x_i}{||x_j-x_i||} - k_3 \dot{x}_i$$

$$U = \ddot{X} = \ddot{X_1} + \ddot{X_2}$$

Where $K$ is a matrix of spring constants, defined as the adjacency matrix for the randomly-generated isomorphic subgraph $T_1$, and $L$ is the natural spring length, set to be the relative distance between two agents in the final target formation $X_{tgt}$.

<img src="https://github.com/ericjhkim/flocking_sandbox/blob/main/visualizations/laplacian_pinning_spring/anim_20250315_154650.gif" style="width:75%;">

### Neuroevolutionary Connectivity Maintenance ([evolve.py -> simulate.py](/evolution/evolve.py))
Here, the Neuroevolution of Augmenting Topologies (NEAT) algorithm [2] is used to train an artificial neural network (ANN) controller that can be homogeneously and ubiquitously applied to agents in a swarm. The ANN evolves based on the fitness function $f$ described below with gains $k$, swarm size $N$, epochs $T$, position for agent $i$ at epoch $t$ being $X^i_t$, and maximum control value $U_{max}$. The fitness minimizes connectivity error $e_c$, achieve position $e_p$ and velocity $e_v$ tracking, and minimize control effort $e_u$.

$$f = -(k_p e_p + k_v e_v + k_c e_c + k_u e_u)$$

$$e_p = \frac{1}{N}\sum^N_{i=1}\left(\frac{||X^i_T - X^i_{tgt}||}{||X^i_0 - X^i_{tgt}||}\right)^2$$

$$e_v = \frac{1}{N}\sum^N_{i=1}||V^i_T||^2$$

$$e_u = \frac{1}{N}\sum^N_{i=1}\left(\frac{1}{T}\sum^T_{t=0}\left(\frac{||U^i_N||}{||U_{max}||}\right)\right)^2$$

$$e_c = {1\times 10^7}^2\text{ if connectivity broken at any epoch; else }0$$

<img src="https://github.com/ericjhkim/flocking_sandbox/blob/main/visualizations/evolution/anim_20250320_163638.gif" style="width:75%;">

## References
  1. Hamaoui, M. (2024). *Connectivity Maintenance through Unlabeled Spanning Tree Matching*. J Intell Robot Syst 110, 15 [doi:10.1007/s10846-024-02048-9](https://doi.org/10.1007/s10846-024-02048-9)
  2. Stanley, K. O., & Miikkulainen, R. (2002). *Evolving neural networks through augmenting topologies*. Evolutionary computation, 10(2), 99-127. [doi:10.1162/106365602320169811](https://doi.org/10.1162/106365602320169811)
