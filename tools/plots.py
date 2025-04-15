"""
This module contains functions for plotting and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import matplotlib.animation as animation
from datetime import datetime
import networkx as nx
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

#%% Plotting and Visualization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
from datetime import datetime

def animate_3d(flock, CREATE_GIF, gif_path, interval=100):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    position = np.array(flock.data["position"])
    N_AGENTS = flock.N_AGENTS
    has_pose = "pose" in flock.data
    if has_pose:
        pose = np.array(flock.data["pose"])  # rotation vectors

    x_min, y_min, z_min = np.min(position, axis=(0, 1))
    x_max, y_max, z_max = np.max(position, axis=(0, 1))

    ax.scatter(flock.X_tgt[:, 0], flock.X_tgt[:, 1], flock.X_tgt[:, 2], color='red', marker='x', label='Target')
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    tails = [ax.plot([], [], [], color='blue', alpha=0.5)[0] for _ in range(N_AGENTS)]
    texts = [ax.text(0, 0, 0, str(i), color='black') for i in range(N_AGENTS)]
    connections = [ax.plot([], [], [], 'k--', alpha=0.3)[0] for _ in range(N_AGENTS ** 2)]
    subgraph_connections = [ax.plot([], [], [], 'k--', alpha=0.3)[0] for _ in range(N_AGENTS ** 2)]

    if has_pose:
        prisms = [Poly3DCollection([], color='blue', alpha=1.0) for _ in range(N_AGENTS)]
        for prism in prisms:
            ax.add_collection3d(prism)
    else:
        scatters = [ax.scatter([], [], [], color='blue') for _ in range(N_AGENTS)]

    def create_prism_at(pos, rotvec, scale=1.0, height=0.1):
        base = np.array([
            [scale, 0, 0],
            [-scale * 0.3, 0, scale * 0.3],
            [-scale * 0.3, 0, -scale * 0.3]
        ])
        top = base + np.array([0, height, 0])
        vertices = np.vstack((base, top))
        R_matrix = R.from_rotvec(rotvec).as_matrix()
        rotated = vertices @ R_matrix.T + pos
        return [
            [rotated[0], rotated[1], rotated[2]],
            [rotated[3], rotated[4], rotated[5]],
            [rotated[0], rotated[1], rotated[4], rotated[3]],
            [rotated[1], rotated[2], rotated[5], rotated[4]],
            [rotated[2], rotated[0], rotated[3], rotated[5]],
        ]

    def init():
        artists = []
        for tail in tails:
            tail.set_data([], [])
            tail.set_3d_properties([])
            artists.append(tail)
        for text in texts:
            text.set_position((0, 0))
            text.set_3d_properties(0)
            artists.append(text)
        for line in connections + subgraph_connections:
            line.set_data([], [])
            line.set_3d_properties([])
            artists.append(line)
        if has_pose:
            for prism in prisms:
                prism.set_verts([])
                artists.append(prism)
        else:
            for scatter in scatters:
                scatter._offsets3d = ([], [], [])
                artists.append(scatter)
        time_text.set_text('')
        artists.append(time_text)
        return artists

    def update(frame):
        adj = flock.data["adjacency"][frame]
        artists = []

        for i in range(N_AGENTS):
            pos = position[frame, i]

            # Tail
            start = max(0, frame - int(1000 / interval))
            tail_pos = position[start:frame+1, i]
            tails[i].set_data(tail_pos[:, 0], tail_pos[:, 1])
            tails[i].set_3d_properties(tail_pos[:, 2])
            artists.append(tails[i])

            # Label
            texts[i].set_position((pos[0], pos[1]))
            texts[i].set_3d_properties(pos[2] + 1.0)
            artists.append(texts[i])

            # Pose logic
            if has_pose:
                rotvec = pose[frame, i]
                faces = create_prism_at(pos, rotvec)
                prisms[i].set_verts(faces)
                artists.append(prisms[i])
            else:
                scatters[i]._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
                artists.append(scatters[i])

        # Subgraph connections
        k = 0
        for i in range(N_AGENTS):
            for j in range(i + 1, N_AGENTS):
                line = subgraph_connections[k]
                if flock.A_T1[i, j] == 1:
                    p1 = position[frame, i]
                    p2 = position[frame, j]
                    line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                    line.set_3d_properties([p1[2], p2[2]])
                    line.set_color('green' if adj[i, j] == 1 else 'red')
                else:
                    line.set_data([], [])
                    line.set_3d_properties([])
                artists.append(line)
                k += 1

        # Extra edges
        k = 0
        for i in range(N_AGENTS):
            for j in range(i + 1, N_AGENTS):
                line = connections[k]
                if adj[i, j] == 1 and flock.A_T1[i, j] == 0:
                    p1 = position[frame, i]
                    p2 = position[frame, j]
                    line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                    line.set_3d_properties([p1[2], p2[2]])
                else:
                    line.set_data([], [])
                    line.set_3d_properties([])
                artists.append(line)
                k += 1

        time_text.set_text(f"Time: {frame * interval / 1000:.2f} s")
        artists.append(time_text)
        return artists

    num_frames = len(position)
    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init,
                                  interval=interval, blit=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    buffer = 2
    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)
    ax.set_zlim(z_min - buffer, z_max + buffer)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = gif_path + f"anim_{timestamp}.gif"
    if CREATE_GIF:
        ani.save(path, writer='pillow', fps=1000/interval)

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
