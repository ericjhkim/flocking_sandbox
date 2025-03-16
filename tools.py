import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import matplotlib.animation as animation
from datetime import datetime
import networkx as nx

#%% Plotting and Visualization
def animate_3d(flock, CREATE_GIF, gif_path, interval=100):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    flock.pos = np.array(flock.pos)

    # Create scatter plot for each agent with the same color (blue)
    scatters = [ax.scatter([], [], [], color='blue') for i in range(flock.N_AGENTS)]

    # Plot target positions with red 'x' markers
    ax.scatter(flock.X_tgt[:, 0], flock.X_tgt[:, 1], flock.X_tgt[:, 2], color='red', marker='x', label='Target')

    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    # Initialize lines for tails
    tails = [ax.plot([], [], [], color='blue', alpha=0.5)[0] for _ in range(flock.N_AGENTS)]

    # Initialize text for agent numbers
    texts = [ax.text(0, 0, 0, str(i), color='black', rotation=90) for i in range(flock.N_AGENTS)]

    # Initialize lines for connectivity
    connections = [ax.plot([], [], [], 'k--', alpha=0.3)[0] for _ in range(flock.N_AGENTS * flock.N_AGENTS)]

    # Initialize lines for subgraph connectivity
    subgraph_connections = [ax.plot([], [], [], 'k--', alpha=0.3)[0] for _ in range(flock.N_AGENTS * flock.N_AGENTS)]

    def init():
        for scatter in scatters:
            scatter._offsets3d = ([], [], [])
        for tail in tails:
            tail.set_data([], [])
            tail.set_3d_properties([])
        for text in texts:
            text.set_position((0, 0))
            text.set_3d_properties(0, 'z')
        for connection in connections:
            connection.set_data([], [])
            connection.set_3d_properties([])
        for subgraph_connection in subgraph_connections:
            subgraph_connection.set_data([], [])
            subgraph_connection.set_3d_properties([])
        time_text.set_text('')
        return scatters + tails + texts + connections + subgraph_connections + [time_text]

    def update(frame):
        adjacency_matrix = flock.get_adjacency(flock.pos[frame])

        for i in range(flock.N_AGENTS):
            positions = flock.pos[frame, i]
            scatters[i]._offsets3d = ([positions[0]], [positions[1]], [positions[2]])

            # Update tail
            start_frame = max(0, frame - int(1000 / interval))
            tail_positions = flock.pos[start_frame:frame+1, i]
            tails[i].set_data(tail_positions[:, 0], tail_positions[:, 1])
            tails[i].set_3d_properties(tail_positions[:, 2])

            # Update text position
            texts[i].set_position((positions[0], positions[1]))
            texts[i].set_3d_properties(positions[2] + 0.1, 'z')  # Slightly above the agent

        # Update subgraph connectivity lines
        subgraph_connection_index = 0
        for i in range(flock.N_AGENTS):
            for j in range(i + 1, flock.N_AGENTS):
                if flock.A_T1[i, j] == 1:               # If there is supposed to be a connection according to the subgraph
                    if adjacency_matrix[i, j] == 1:     # If there actually is a connection
                        color = 'green'
                    else:
                        color = 'red'                   # If there is no actual connection even when there is supposed to be one
                    pos_i = flock.pos[frame, i]
                    pos_j = flock.pos[frame, j]
                    subgraph_connections[subgraph_connection_index].set_data([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]])
                    subgraph_connections[subgraph_connection_index].set_3d_properties([pos_i[2], pos_j[2]])
                    subgraph_connections[subgraph_connection_index].set_color(color)
                    subgraph_connections[subgraph_connection_index].set_alpha(0.3)
                else:                                   # Ignore if there isn't even supposed to be a connection
                    subgraph_connections[subgraph_connection_index].set_data([], [])
                    subgraph_connections[subgraph_connection_index].set_3d_properties([])
                subgraph_connection_index += 1

        # Update connectivity lines
        connection_index = 0
        for i in range(flock.N_AGENTS):
            for j in range(i + 1, flock.N_AGENTS):
                if adjacency_matrix[i, j] == 1 and flock.A_T1[i, j] == 0:
                    pos_i = flock.pos[frame, i]
                    pos_j = flock.pos[frame, j]
                    connections[connection_index].set_data([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]])
                    connections[connection_index].set_3d_properties([pos_i[2], pos_j[2]])
                else:
                    connections[connection_index].set_data([], [])
                    connections[connection_index].set_3d_properties([])
                connection_index += 1

        time_text.set_text(f'Time: {frame * interval / 1000:.2f} s')
        return scatters + tails + texts + connections + subgraph_connections + [time_text]

    num_frames = len(flock.pos)

    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init,
                                  interval=interval, blit=False)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.tight_layout()
    plt.gca().set_aspect('equal')

    if CREATE_GIF:
        ani.save(gif_path, writer='pillow', fps=1000/interval)

    plt.show()

def draw_graph(nx_graph):
    """
    Visualizes a single graph.
    """
    fig, axes = plt.subplots(1,1,dpi=72)
    nx.draw(nx_graph, pos=nx.spring_layout(nx_graph), ax=axes, with_labels=True)
    plt.tight_layout()
    plt.show()

def draw_graphs(graphs):
    """
    Visualizes multiple graphs in a single plot.
    """
    num_graphs = len(graphs)
    fig, axes = plt.subplots(1, num_graphs, dpi=72, figsize=(5 * num_graphs, 5))  

    if num_graphs == 1:  # Handle single graph case
        axes = [axes]  

    colors = ["#AEC6CF", "#77DD77", "#FFB6C1", "#D8BFD8", "#FFDAB9", "#C4A484", "#F4C2C2", "#AFEEEE"]
    titles = ["G1","T1","G2","T2"]

    for g, graph in enumerate(graphs):
        pos = nx.spring_layout(graph)  # Compute positions
        nx.draw(graph, pos, ax=axes[g], with_labels=True, node_color=colors[g % len(colors)], edge_color="gray")
        axes[g].set_title(titles[g], fontsize=12, fontweight="bold")  

    plt.tight_layout()
    plt.show()
