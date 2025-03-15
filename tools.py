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

    flock.data = np.array(flock.data)

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
        adjacency_matrix = flock.get_adjacency(flock.data[frame])

        for i in range(flock.N_AGENTS):
            positions = flock.data[frame, i]
            scatters[i]._offsets3d = ([positions[0]], [positions[1]], [positions[2]])

            # Update tail
            start_frame = max(0, frame - int(1000 / interval))
            tail_positions = flock.data[start_frame:frame+1, i]
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
                    pos_i = flock.data[frame, i]
                    pos_j = flock.data[frame, j]
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
                    pos_i = flock.data[frame, i]
                    pos_j = flock.data[frame, j]
                    connections[connection_index].set_data([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]])
                    connections[connection_index].set_3d_properties([pos_i[2], pos_j[2]])
                else:
                    connections[connection_index].set_data([], [])
                    connections[connection_index].set_3d_properties([])
                connection_index += 1

        time_text.set_text(f'Time: {frame * interval / 1000:.2f} s')
        return scatters + tails + texts + connections + subgraph_connections + [time_text]

    num_frames = len(flock.data)

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

def plot_data(data, initial, target):
    """
    Plots the position and velocity of drone agents from a 3D NumPy array.
    
    Parameters:
    - data: A 3D NumPy array of shape (N_TIME_STEPS, 4, N_AGENTS) where each 4xN_AGENTS sub-array
            represents the position (rows 0-2) and velocity (rows 2-4) of N_AGENTS drone agents.
    """
    num_time_steps = data.shape[0]
    num_drones = data.shape[2]  # Number of drone agents

    # Create subplots for position and velocity
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot positions
    for drone_id in range(num_drones):
        positions = data[:, 0:2, drone_id]  # Extract x and y positions over time
        axes[0].plot(positions[:, 0], positions[:, 1], label=f'Drone {drone_id + 1}')
    
    axes[0].set_title('Drone Positions Over Time')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # Plot initial and target coordinates
    for drone_id in range(num_drones):
        positions_i = initial[:2, drone_id]
        positions_f = target[:2, drone_id]
        axes[0].scatter(positions_i[0], positions_i[1], c="g", marker="o")
        axes[0].scatter(positions_f[0], positions_f[1], c="r", marker="o")
    
    # Plot velocities
    for drone_id in range(num_drones):
        velocities = data[:, 2:4, drone_id]  # Extract x and y velocities over time
        axes[1].plot(np.arange(num_time_steps), velocities[:, 0], label=f'Drone {drone_id + 1} X-Velocity')
        axes[1].plot(np.arange(num_time_steps), velocities[:, 1], linestyle='--', label=f'Drone {drone_id + 1} Y-Velocity')
    
    axes[1].set_title('Drone Velocities Over Time')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Velocity')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_agents_3d(coords, A):
    """
    Visualizes the positions of agents and their connections in a 3D plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]

    # Plot the points with labels
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ax.scatter(xi, yi, zi, c='b', marker='o')
        ax.text(xi, yi, zi, f'{i}', color='red')

    # Draw connections based on adjacency matrix
    N = len(x)
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] == 1:
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c='gray', linestyle='--')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('3D Agent Positions with Connections')

    plt.tight_layout()
    plt.show()

def plot_agents_connectivity(coords, A, A_tgt):
    """
    Plots the connections between agents and the validity of connections against a target adjacency matrix reference.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates
    x = coords[0,:]
    y = coords[1,:]
    z = coords[2,:]

    # Plot the points with labels
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ax.scatter(xi, yi, zi, c='b', marker='o')
        ax.text(xi, yi, zi, f'{i}', color='red')

    N_AGENTS = len(x)
    ## Plot edges
    for i in range(N_AGENTS):
        for j in range(N_AGENTS):
            if i != j:
                color = None
                if A_tgt[i, j] == 1 and A[i, j] == 1:
                    color = "green"
                elif A_tgt[i, j] == 0 and A[i, j] == 1:
                    color = "gray"
                elif A_tgt[i, j] == 1 and A[i, j] == 0:
                    color = "red"

                if color is not None:
                    x_values = [coords[0, i], coords[0, j]]
                    y_values = [coords[1, i], coords[1, j]]
                    z_values = [coords[2, i], coords[2, j]]
                    ax.plot(x_values, y_values, z_values, color=color, linewidth=1.0, zorder=1)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('3D Agent Positions with Connections')

    plt.tight_layout()
    plt.show()

def plot_separation(data):
    """
    Plots the average, minimum, and maximum separation distances among agents.

    Parameters:
        data: A t x d x i array where:
              t - number of timesteps,
              d - number of dimensions (4 or 6),
              i - number of agents.
    """
    # Ensure valid shape
    if data.ndim != 3:
        raise ValueError("Data must be a t x d x i array.")
    
    t, d, i = data.shape
    if d not in [4, 6]:
        raise ValueError("The dimension d must be either 4 or 6.")
    
    # Extract position data (first 2 or 3 dimensions)
    positions = data[:, :3 if d == 6 else 2, :]  # Shape: t x (2 or 3) x i
    
    # Compute pairwise distances
    distances = np.linalg.norm(
        positions[:, :, :, None] - positions[:, :, None, :], axis=1
    )  # Shape: t x i x i
    
    # Mask diagonal elements (self-distance)
    np.fill_diagonal(distances[0], np.nan)  # Mask diagonal on a sample timestep
    for step in range(1, t):
        np.fill_diagonal(distances[step], np.nan)
    
    # Compute statistics
    avg_separation = np.nanmean(distances, axis=(1, 2))
    min_separation = np.nanmin(distances, axis=(1, 2))
    max_separation = np.nanmax(distances, axis=(1, 2))
    
    # Plot results
    timesteps = np.arange(t)
    
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, avg_separation, label="Average Separation", color="blue")
    plt.fill_between(
        timesteps, min_separation, max_separation, color="lightblue", alpha=0.5,
        label="Range (Min to Max)"
    )
    plt.title("Separation Distance Analysis")
    plt.xlabel("Time Steps")
    plt.ylabel("Separation Distance")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def get_lims(states):
    """
    Computes the axis limits across the entire trajectory.
    """
    lims = []
    for i in range(3):
        min_value = min(min(states_t[i]) for states_t in states)
        max_value = max(max(states_t[i]) for states_t in states)
        buffer = max_value*0.1
        lims.append([min_value-buffer, max_value+buffer])
    
    return lims

def create_frame(states, A, A_st, t, frame_count, last, initial, final, lims, F=None):
    """
    Create a single frame for the GIF animation.
    """
    # Turn off interactive mode
    plt.ioff()

    def create_prism_points_3d(x, y, z, vx, vy, vz, scale=1.0):
        """Generate 3D triangular prism points oriented in the direction of the velocity vector."""
        direction_vector = np.array([vx, vy, vz])
        norm = np.linalg.norm(direction_vector)
        if norm == 0:
            return np.array([[x, y, z]])  # Return a single point if velocity is zero

        # Normalize the velocity vector
        direction_vector /= norm

        # Create triangle shape in the XZ plane with the nose pointing in the +x direction
        triangle_base = np.array([[scale, 0, 0], [-scale * 0.5, 0, scale * 0.5], [-scale * 0.5, 0, -scale * 0.5]])

        # Create rotation matrix to align x-axis with direction vector
        x_axis = np.array([1, 0, 0])
        v = np.cross(x_axis, direction_vector)
        s = np.linalg.norm(v)
        c = np.dot(x_axis, direction_vector)

        if s != 0:
            vx_matrix = np.array([[0, -v[2], v[1]],
                                [v[2], 0, -v[0]],
                                [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + vx_matrix + (vx_matrix @ vx_matrix) * ((1 - c) / (s ** 2))
            rotated_triangle = (rotation_matrix @ triangle_base.T).T
        else:
            rotated_triangle = triangle_base

        return rotated_triangle + np.array([x, y, z])
    
    N_AGENTS = states.shape[1]

    fig = plt.figure(figsize=(5,4), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_zlim(lims[2])
    
    # Set axis labels
    ax.set_title(f'Simulation Time: {t:.2f}s')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')

    ## Plot edges
    for i in range(N_AGENTS):
        for j in range(N_AGENTS):
            if i != j:
                color = None
                if A_st[i, j] == 1 and A[i, j] == 1:
                    color = "green"
                elif A_st[i, j] == 0 and A[i, j] == 1:
                    color = "gray"
                elif A_st[i, j] == 1 and A[i, j] == 0:
                    color = "red"

                if color is not None:
                    x_values = [states[0, i], states[0, j]]
                    y_values = [states[1, i], states[1, j]]
                    z_values = [states[2, i], states[2, j]]
                    ax.plot(x_values, y_values, z_values, color=color, linewidth=0.5, zorder=1)

    ## Plot agents  (triangle polygons)
    for i in range(N_AGENTS):
        x, y, z = states[0, i], states[1, i], states[2, i]
        vx, vy, vz = states[3, i], states[4, i], states[5, i]
        prism_points = create_prism_points_3d(x, y, z, vx, vy, vz, scale=1.5)

        poly = art3d.Poly3DCollection([prism_points], color="black")
        ax.add_collection3d(poly)

    ## Draw initial and final positions
    # Extract x, y, z coordinates
    x = initial[0,:]
    y = initial[1,:]
    z = initial[2,:]
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ax.scatter(xi, yi, zi, c='b', marker='o', s=4)

    # Extract x, y, z coordinates
    x = final[0,:]
    y = final[1,:]
    z = final[2,:]
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ax.scatter(xi, yi, zi, c='r', marker='o', s=4)

    # Save the current frame as an image
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f'frames/frame_{frame_count:04d}.png')
    if last:
        plt.show()
    plt.close(fig)  # Close the figure to prevent it from displaying

#%% Data Management
def save_to_h5py(data, filename="sim_data", dataset_name="data"):
    """
    Saves a numpy array to an HDF5 file using h5py, with a timestamp in the filename.

    Parameters:
        data (numpy.ndarray): The data array to save (t x d x i).
        filename (str): Base name of the file (default: 'sim_data').
        dataset_name (str): Name of the dataset inside the HDF5 file (default: 'data').
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    
    # Generate a timestamp
    filename = f"{filename}.h5"

    with h5py.File(filename, "w") as h5file:
        h5file.create_dataset(dataset_name, data=data)
    
    print(f"Data saved to {filename} under dataset '{dataset_name}'.")

def load_latest_h5py(filename="sim_data", dataset_name="data", directory="data"):
    """
    Loads the most recent HDF5 file based on the timestamp in the filename.

    Parameters:
        filename (str): Base name of the file to look for (default: 'sim_data').
        dataset_name (str): Name of the dataset inside the HDF5 file (default: 'data').
        directory (str): Directory to search for the files (default: 'data').

    Returns:
        numpy.ndarray: The loaded data array from the latest file.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    
    # Compile a regex to match files with timestamps
    pattern = re.compile(f"{re.escape(filename)}_(\\d{{8}}_\\d{{6}})\\.h5$")
    
    # List all files in the specified directory
    files = os.listdir(directory)
    
    # Filter and extract valid timestamped files
    timestamped_files = []
    for file in files:
        match = pattern.match(file)
        if match:
            timestamped_files.append((file, match.group(1)))
    
    if not timestamped_files:
        raise FileNotFoundError(f"No files matching the pattern {filename}_<timestamp>.h5 found in '{directory}'.")
    
    # Sort files by timestamp
    timestamped_files.sort(key=lambda x: datetime.strptime(x[1], "%Y%m%d_%H%M%S"), reverse=True)
    latest_file = timestamped_files[0][0]
    
    # Load the latest file
    file_path = os.path.join(directory, latest_file)
    with h5py.File(file_path, "r") as h5file:
        data = h5file[dataset_name][:]
    
    print(f"Loaded data from {file_path} under dataset '{dataset_name}'.")
    return data