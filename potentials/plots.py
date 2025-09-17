"""
This module contains functions for plotting and visualization.
This is a stripped copy of the tools.plots.py file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import proj3d

def animate_3d(
        flock,
        CREATE_GIF,
        gif_path,
        equal_aspect=True,              # set equal aspect ratio
        interval=100,                   # milliseconds between frames
        follow=False,                   # off by default for backward compatibility
        follow_padding=2.0,             # padding around the tight bounds (units of your sim)
        follow_smooth=None,             # None = no smoothing; or set to 0<alpha<=1 (EMA)
        draw_walls=False,               # draw environment walls if available
        env=None,                       # explicit env; if None, tries flock.env
    ):
    fig = plt.figure(figsize=(16, 8))
    ax3d = fig.add_subplot(121, projection='3d')  # 3D plot on the left
    ax2d = fig.add_subplot(122)                   # 2D plot on the right

    # 2D plot ------------------------------------------------------------------------------------
    x = np.arange(0, 10, 0.01)
    y = flock.potential.get_function(x)

    r0 = flock.potential.params.get('r0', 1.0)
    ax2d.plot(r0, flock.potential.get_function(r0), 'ro', label='Equilibrium Point', zorder=10)
    ax2d.plot(x, y, label='Potential $\phi(r)$')

    inter_agent_point = ax2d.scatter([], [], color='black', zorder=12)

    ax2d.set_ylim(-2.0, 10.0)

    ax2d.set_xlabel('Inter-Agent Distance $r$')
    ax2d.set_ylabel('Value')
    ax2d.legend()
    ax2d.grid(True)

    # 3D plot ------------------------------------------------------------------------------------
    position = np.array(flock.data["position"])  # shape: (T, N, 3)
    N_AGENTS = flock.N_AGENTS

    has_pose = "pose" in flock.data
    if has_pose:
        pose = np.array(flock.data["pose"])  # rotation vectors, shape: (T, N, 3)

    # Static world bounds (used if follow=False or as initial/safety bounds)
    x_min, y_min, z_min = np.min(position, axis=(0, 1))
    x_max, y_max, z_max = np.max(position, axis=(0, 1))

    if hasattr(flock, 'X_tgt'):
        ax3d.scatter(flock.X_tgt[:, 0], flock.X_tgt[:, 1], flock.X_tgt[:, 2], color='red', marker='x', label='Target')

    time_text = ax3d.text2D(0.05, 0.95, '', transform=ax3d.transAxes)
    distance_text = ax3d.text2D(0.5, 0.5, '', transform=ax3d.transAxes, color='black', fontsize=14, zorder=20)

    tails = [ax3d.plot([], [], [], color='blue', alpha=0.5)[0] for _ in range(N_AGENTS)]
    texts = [ax3d.text(0, 0, 0, str(i), color='black') for i in range(N_AGENTS)]
    connections = [ax3d.plot([], [], [], 'k--', alpha=0.3)[0] for _ in range(N_AGENTS ** 2)]
    subgraph_connections = [ax3d.plot([], [], [], 'k--', alpha=0.3)[0] for _ in range(N_AGENTS ** 2)]

    if has_pose:
        prisms = [Poly3DCollection([], color='blue', alpha=1.0) for _ in range(N_AGENTS)]
        for prism in prisms:
            ax3d.add_collection3d(prism)
    else:
        scatters = [ax3d.scatter([], [], [], color='blue') for _ in range(N_AGENTS)]

    # === Helpers for prisms ===
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

    # === Camera-follow state (for smoothing) ===
    if follow and (follow_smooth is not None):
        alpha = float(follow_smooth)
        alpha = max(0.0, min(1.0, alpha))
    else:
        alpha = None

    def tight_bounds_for_frame(f):
        p = position[f]  # shape (N, 3)
        finite = np.isfinite(p).all(axis=1)
        if not np.any(finite):
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

    if follow:
        tb = tight_bounds_for_frame(0)
        sxmin, sxmax, symin, symax, szmin, szmax = tb
    else:
        sxmin, sxmax, symin, symax, szmin, szmax = x_min, x_max, y_min, y_max, z_min, z_max

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

        distance_text.set_position((0, 0))
        distance_text.set_text('')
        artists.append(distance_text)

        inter_agent_point.set_offsets(np.empty((0, 2)))
        artists.append(inter_agent_point)

        time_text.set_text('')
        artists.append(time_text)

        # Set initial limits
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        ax3d.set_xlim(sxmin, sxmax)
        ax3d.set_ylim(symin, symax)
        ax3d.set_zlim(szmin, szmax)
        return artists

    def update(frame):
        nonlocal sxmin, sxmax, symin, symax, szmin, szmax

        adj = flock.data["adjacency"][frame]
        artists = []

        # Draw agents
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

            # Pose or scatter
            if has_pose:
                rotvec = pose[frame, i]
                faces = create_prism_at(pos, rotvec)
                prisms[i].set_verts(faces)
                artists.append(prisms[i])
            else:
                scatters[i]._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
                artists.append(scatters[i])

        # Update inter-agent distance point on 2D plot
        X_i = position[frame, 0]
        X_j = position[frame, 1]
        r_ij = np.linalg.norm(X_i - X_j)
        phi_ij = flock.potential.get_function(r_ij)
        inter_agent_point.set_offsets([[r_ij, phi_ij]])
        artists.append(inter_agent_point)

        # # Update inter-agent distance text (only for 2 agents)
        # Calculate midpoint in 3D
        midpoint = (X_i + X_j) / 2

        # Project 3D midpoint to 2D display coordinates
        x2, y2, _ = proj3d.proj_transform(midpoint[0]+3, midpoint[1], midpoint[2], ax3d.get_proj())
        trans = ax3d.transData.transform((x2, y2))
        inv = fig.transFigure.inverted()
        x_fig, y_fig = inv.transform(trans)

        # Update the text position and value
        distance_text.set_position((x_fig, y_fig))
        distance_text.set_text(f"{r_ij:.2f}")
        artists.append(distance_text)

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

        # Update time text
        time_text.set_text(f"Time: {frame * interval / 1000:.2f} s")
        artists.append(time_text)

        # === Camera follow ===
        if follow:
            txmin, txmax, tymin, tymax, tzmin, tzmax = tight_bounds_for_frame(frame)

            if alpha is None:
                sxmin, sxmax = txmin, txmax
                symin, symax = tymin, tymax
                szmin, szmax = tzmin, tzmax
            else:
                sxmin = alpha * txmin + (1 - alpha) * sxmin
                sxmax = alpha * txmax + (1 - alpha) * sxmax
                symin = alpha * tymin + (1 - alpha) * symin
                symax = alpha * tymax + (1 - alpha) * symax
                szmin = alpha * tzmin + (1 - alpha) * szmin
                szmax = alpha * tzmax + (1 - alpha) * szmax

            ax3d.set_xlim(sxmin, sxmax)
            ax3d.set_ylim(symin, symax)
            ax3d.set_zlim(szmin, szmax)

        return artists

    num_frames = len(position)
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, interval=interval, blit=False
    )

    # Static limits if not following
    if not follow:
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        buffer = 2
        ax3d.set_xlim(x_min - buffer, x_max + buffer)
        ax3d.set_ylim(y_min - buffer, y_max + buffer)
        ax3d.set_zlim(z_min - buffer, z_max + buffer)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = gif_path + f"anim_{timestamp}.gif"
    if CREATE_GIF:
        ani.save(path, writer='pillow', fps=1000/interval)

    if equal_aspect:
        ax3d.set_aspect('equal')
    plt.show()
