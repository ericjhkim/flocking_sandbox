"""
This module contains functions for plotting and visualization.
This is a stripped copy of the tools.plots.py file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def animate_3d(
        flocks,
        CREATE_GIF,
        gif_path,
        equal_aspect=True,
        interval=100,
        follow=False,
        follow_padding=2.0,
        follow_smooth=None,
        draw_walls=False,
        env=None,
    ):
    fig = plt.figure(figsize=(12, 5))
    ax3d = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]

    # Prepare data for both flocks
    positions = [np.array(flock.data["position"]) for flock in flocks]
    N_AGENTS = [flock.N_AGENTS for flock in flocks]
    has_pose = ["pose" in flock.data for flock in flocks]

    # World bounds for both flocks
    bounds = []
    for pos in positions:
        x_min, y_min, z_min = np.min(pos, axis=(0, 1))
        x_max, y_max, z_max = np.max(pos, axis=(0, 1))
        bounds.append((x_min, x_max, y_min, y_max, z_min, z_max))

    # Target positions
    for idx, flock in enumerate(flocks):
        if hasattr(flock, 'X_tgt'):
            ax3d[idx].scatter(flock.X_tgt[:, 0], flock.X_tgt[:, 1], flock.X_tgt[:, 2], color='red', marker='x', label='Target')

    # Artists for both axes
    time_text = [ax3d[i].text2D(0.05, 0.95, '', transform=ax3d[i].transAxes) for i in range(2)]
    tails = [[ax3d[i].plot([], [], [], color='blue', alpha=0.5)[0] for _ in range(N_AGENTS[i])] for i in range(2)]
    # texts = [[ax3d[i].text(0, 0, 0, str(j), color='black') for j in range(N_AGENTS[i])] for i in range(2)]
    connections = [[ax3d[i].plot([], [], [], 'k--', alpha=0.3)[0] for _ in range(N_AGENTS[i] ** 2)] for i in range(2)]
    subgraph_connections = [[ax3d[i].plot([], [], [], 'k--', alpha=0.3)[0] for _ in range(N_AGENTS[i] ** 2)] for i in range(2)]

    prisms = []
    scatters = []
    for idx in range(2):
        if has_pose[idx]:
            prisms.append([Poly3DCollection([], color='blue', alpha=1.0) for _ in range(N_AGENTS[idx])])
            for prism in prisms[idx]:
                ax3d[idx].add_collection3d(prism)
            scatters.append(None)
        else:
            scatters.append([ax3d[idx].scatter([], [], [], color='blue') for _ in range(N_AGENTS[idx])])
            prisms.append(None)

    # Camera-follow state
    if follow and (follow_smooth is not None):
        alpha = float(follow_smooth)
        alpha = max(0.0, min(1.0, alpha))
    else:
        alpha = None

    def tight_bounds_for_frame(pos, f, follow_padding):
        p = pos[f]
        finite = np.isfinite(p).all(axis=1)
        if not np.any(finite):
            x_min, x_max, y_min, y_max, z_min, z_max = np.min(pos, axis=(0, 1)), np.max(pos, axis=(0, 1)), np.min(pos, axis=(0, 1)), np.max(pos, axis=(0, 1)), np.min(pos, axis=(0, 1)), np.max(pos, axis=(0, 1))
            return (x_min, x_max, y_min, y_max, z_min, z_max)
        p = p[finite]
        t_xmin, t_ymin, t_zmin = np.min(p, axis=0)
        t_xmax, t_ymax, t_zmax = np.max(p, axis=0)
        pad = float(follow_padding)
        eps = 1e-6
        if t_xmax - t_xmin < eps:
            t_xmin -= pad * 0.5
            t_xmax += pad * 0.5
        if t_ymax - t_ymin < eps:
            t_ymin -= pad * 0.5
            t_ymax += pad * 0.5
        if t_zmax - t_zmin < eps:
            t_zmin -= pad * 0.5
            t_zmax += pad * 0.5
        return (t_xmin - pad, t_xmax + pad, t_ymin - pad, t_ymax + pad, t_zmin - pad, t_zmax + pad)

    # Initial bounds
    sxmin, sxmax, symin, symax, szmin, szmax = [b[0] for b in bounds], [b[1] for b in bounds], [b[2] for b in bounds], [b[3] for b in bounds], [b[4] for b in bounds], [b[5] for b in bounds]

    def init():
        artists = []
        for idx in range(2):
            for tail in tails[idx]:
                tail.set_data([], [])
                tail.set_3d_properties([])
                artists.append(tail)
            # for text in texts[idx]:
            #     text.set_position((0, 0))
            #     text.set_3d_properties(0)
            #     artists.append(text)
            for line in connections[idx] + subgraph_connections[idx]:
                line.set_data([], [])
                line.set_3d_properties([])
                artists.append(line)
            if has_pose[idx]:
                for prism in prisms[idx]:
                    prism.set_verts([])
                    artists.append(prism)
            else:
                for scatter in scatters[idx]:
                    scatter._offsets3d = ([], [], [])
                    artists.append(scatter)
            time_text[idx].set_text('')
            artists.append(time_text[idx])
            ax3d[idx].set_xlabel('X')
            ax3d[idx].set_ylabel('Y')
            ax3d[idx].set_zlabel('Z')
            ax3d[idx].set_xlim(sxmin[idx], sxmax[idx])
            ax3d[idx].set_ylim(symin[idx], symax[idx])
            ax3d[idx].set_zlim(szmin[idx], szmax[idx])
        return artists

    def update(frame):
        artists = []
        for idx in range(2):
            pos = positions[idx]
            flock = flocks[idx]
            adj = flock.data["adjacency"][frame]
            # Draw agents
            for i in range(N_AGENTS[idx]):
                p = pos[frame, i]
                # Tail
                start = max(0, frame - int(1000 / interval))
                tail_pos = pos[start:frame+1, i]
                tails[idx][i].set_data(tail_pos[:, 0], tail_pos[:, 1])
                tails[idx][i].set_3d_properties(tail_pos[:, 2])
                artists.append(tails[idx][i])
                # Label
                # texts[idx][i].set_position((p[0], p[1]))
                # texts[idx][i].set_3d_properties(p[2] + 1.0)
                # artists.append(texts[idx][i])
                # Pose or scatter
                scatters[idx][i]._offsets3d = ([p[0]], [p[1]], [p[2]])
                artists.append(scatters[idx][i])
            # Subgraph connections
            k = 0
            for i in range(N_AGENTS[idx]):
                for j in range(i + 1, N_AGENTS[idx]):
                    line = subgraph_connections[idx][k]
                    if flock.A_T1[i, j] == 1:
                        p1 = pos[frame, i]
                        p2 = pos[frame, j]
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
            for i in range(N_AGENTS[idx]):
                for j in range(i + 1, N_AGENTS[idx]):
                    line = connections[idx][k]
                    if adj[i, j] == 1 and flock.A_T1[i, j] == 0:
                        p1 = pos[frame, i]
                        p2 = pos[frame, j]
                        line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                        line.set_3d_properties([p1[2], p2[2]])
                    else:
                        line.set_data([], [])
                        line.set_3d_properties([])
                    artists.append(line)
                    k += 1
            # Update time text
            time_text[idx].set_text(f"Time: {frame * interval / 1000:.2f} s")
            artists.append(time_text[idx])
            # Camera follow
            if follow:
                txmin, txmax, tymin, tymax, tzmin, tzmax = tight_bounds_for_frame(pos, frame, follow_padding)
                if alpha is None:
                    sxmin[idx], sxmax[idx] = txmin, txmax
                    symin[idx], symax[idx] = tymin, tymax
                    szmin[idx], szmax[idx] = tzmin, tzmax
                else:
                    sxmin[idx] = alpha * txmin + (1 - alpha) * sxmin[idx]
                    sxmax[idx] = alpha * txmax + (1 - alpha) * sxmax[idx]
                    symin[idx] = alpha * tymin + (1 - alpha) * symin[idx]
                    symax[idx] = alpha * tymax + (1 - alpha) * symax[idx]
                    szmin[idx] = alpha * tzmin + (1 - alpha) * szmin[idx]
                    szmax[idx] = alpha * tzmax + (1 - alpha) * szmax[idx]
                ax3d[idx].set_xlim(sxmin[idx], sxmax[idx])
                ax3d[idx].set_ylim(symin[idx], symax[idx])
                ax3d[idx].set_zlim(szmin[idx], szmax[idx])
        return artists

    num_frames = min(len(positions[0]), len(positions[1]))
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, interval=interval, blit=False
    )

    # Static limits if not following
    if not follow:
        for idx in range(2):
            ax3d[idx].set_xlabel('X')
            ax3d[idx].set_ylabel('Y')
            ax3d[idx].set_zlabel('Z')
            buffer = 2
            x_min, x_max, y_min, y_max, z_min, z_max = bounds[idx]
            ax3d[idx].set_xlim(x_min - buffer, x_max + buffer)
            ax3d[idx].set_ylim(y_min - buffer, y_max + buffer)
            ax3d[idx].set_zlim(z_min - buffer, z_max + buffer)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = gif_path + f"anim_{timestamp}.gif"
    if CREATE_GIF:
        ani.save(path, writer='pillow', fps=1000/interval)

    if equal_aspect:
        for idx in range(2):
            ax3d[idx].set_aspect('equal')
    plt.show()
