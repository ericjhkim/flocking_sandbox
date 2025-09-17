"""
This module contains functions for plotting and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import networkx as nx
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

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
        walls_alpha=0.25,               # transparency for walls
        walls_color="#555",             # color for walls
        contact_rings=False,            # draw collision circles around agents
        obstacle_rays=False,            # draw sensor rays if available
        x_lines=False,                  # draw +X direction lines on agents
        estimated_neighbours=None,      # plot estimated neighbours with respect to the agent represented by this index
    ):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    position = np.array(flock.data["position"])  # shape: (T, N, 3)
    N_AGENTS = flock.N_AGENTS

    has_pose = "pose" in flock.data
    if has_pose:
        pose = np.array(flock.data["pose"])  # rotation vectors, shape: (T, N, 3)

    if "rays" in flock.data and obstacle_rays:
        rays = np.array(flock.data["rays"])
    else:
        rays = None

    # Static world bounds (used if follow=False or as initial/safety bounds)
    x_min, y_min, z_min = np.min(position, axis=(0, 1))
    x_max, y_max, z_max = np.max(position, axis=(0, 1))

    if hasattr(flock, 'X_tgt'):
        ax.scatter(flock.X_tgt[:, 0], flock.X_tgt[:, 1], flock.X_tgt[:, 2], color='red', marker='x', label='Target')

    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    tails = [ax.plot([], [], [], color='blue', alpha=0.5)[0] for _ in range(N_AGENTS)]
    texts = [ax.text(0, 0, 0, str(i), color='black') for i in range(N_AGENTS)]
    connections = [ax.plot([], [], [], 'k--', alpha=0.3)[0] for _ in range(N_AGENTS ** 2)]
    subgraph_connections = [ax.plot([], [], [], 'k--', alpha=0.3)[0] for _ in range(N_AGENTS ** 2)]
    if x_lines:
        agent_x_lines = [ax.plot([], [], [], color='black')[0] for _ in range(N_AGENTS)]
    if contact_rings:
        collision_circle = [ax.plot([], [], [], color='red', alpha=0.3)[0] for _ in range(N_AGENTS)]
    if rays is not None:
        sensor_rays = [
            [ax.plot([], [], [], color='orange', linewidth=1, linestyle='--', alpha=0.5)[0] for _ in range(flock.K_s)]
            for _ in range(N_AGENTS)
        ]

    if has_pose:
        prisms = [Poly3DCollection([], color='blue', alpha=1.0) for _ in range(N_AGENTS)]
        for prism in prisms:
            ax.add_collection3d(prism)
    else:
        scatters = [ax.scatter([], [], [], color='blue') for _ in range(N_AGENTS)]

    if estimated_neighbours is not None:
        estim_hist = {k: np.array(v) for k, v in flock.data["neighbour_estimates_pos"].items()}
        est_n_scatters = [ax.scatter([], [], [], color='magenta') for _ in range(N_AGENTS)]
        est_n_tails = [ax.plot([], [], [], color='magenta', alpha=0.5)[0] for _ in range(N_AGENTS)]

    # --- WALL SETUP (static geometry) ---
    env_obj = env if env is not None else getattr(flock, "ENV", None)
    wall_patches, wall_objs = [], []

    if draw_walls and (env_obj is not None) and getattr(env_obj, "walls", None):
        init_limits = (x_min, x_max, y_min, y_max, z_min, z_max)
        for wall in env_obj.walls:
            faces = _faces_for_wall_clipped_to_axes(wall, init_limits)
            patch = Poly3DCollection(faces, alpha=walls_alpha, facecolor=walls_color,
                                     edgecolor="k", linewidths=0.5)
            ax.add_collection3d(patch)
            wall_patches.append(patch)
            wall_objs.append(wall)

    def _refresh_walls_to_axes():
        if not wall_patches:
            return
        lims = _axes_limits(ax)
        for patch, wall in zip(wall_patches, wall_objs):
            patch.set_verts(_faces_for_wall_clipped_to_axes(wall, lims))

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

        if x_lines:
            for line in agent_x_lines:
                line.set_data([], [])
                line.set_3d_properties([])
                artists.append(line)
        if contact_rings:
            for line in collision_circle:
                line.set_data([], [])
                line.set_3d_properties([])
                artists.append(line)
        if rays is not None:
            for r in sensor_rays:
                for line in r:
                    line.set_data([], [])
                    line.set_3d_properties([])
                    artists.append(line)

        if estimated_neighbours is not None:
            for scatter in est_n_scatters:
                scatter._offsets3d = ([], [], [])
                artists.append(scatter)
            for esttail in est_n_tails:
                esttail.set_data([], [])
                esttail.set_3d_properties([])
                artists.append(esttail)

        time_text.set_text('')
        artists.append(time_text)

        # Set initial limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(sxmin, sxmax)
        ax.set_ylim(symin, symax)
        ax.set_zlim(szmin, szmax)
        _refresh_walls_to_axes()
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
            
            # Update lines indicating +X direction
            if x_lines:
                start = pos
                end = pos + np.array([2.0, 0, 0])  # length 2 in +X
                agent_x_lines[i].set_data([start[0], end[0]], [start[1], end[1]])
                agent_x_lines[i].set_3d_properties([start[2], end[2]])
                artists.append(agent_x_lines[i])

            # Collision circle
            if contact_rings:
                theta = np.linspace(0, 2 * np.pi, 100)
                pos = position[frame, i]
                x = pos[0] + flock.R_MIN * np.cos(theta)
                y = pos[1] + flock.R_MIN * np.sin(theta)
                z = np.full_like(theta, pos[2])
                collision_circle[i].set_data(x, y)
                collision_circle[i].set_3d_properties(z)
                artists.append(collision_circle[i])

            # Sensor rays
            if rays is not None:
                for k in range(flock.K_s):
                    if rays[frame][i][k] != 0:
                        angle = -2 * np.pi * k / flock.K_s  # Clockwise from +x
                        direction = np.array([np.cos(angle), np.sin(angle), 0])
                        length = rays[frame][i][k]
                        start = pos
                        end = pos + length * direction
                        sensor_rays[i][k].set_data([start[0], end[0]], [start[1], end[1]])
                        sensor_rays[i][k].set_3d_properties([start[2], end[2]])
                    else:
                        # Clear the ray if not active
                        sensor_rays[i][k].set_data([], [])
                        sensor_rays[i][k].set_3d_properties([])
                    artists.append(sensor_rays[i][k])

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

            if estimated_neighbours is not None:
                for j in range(N_AGENTS):
                    if (estimated_neighbours, j) in estim_hist.keys():
                        estpos = estim_hist[(estimated_neighbours, j)]

                        # Scatter
                        est_n_scatters[j]._offsets3d = ([estpos[frame, 0]], [estpos[frame, 1]], [estpos[frame, 2]])
                        artists.append(est_n_scatters[j])

                        # Tails
                        esttail_pos = estpos[start:frame+1]
                        est_n_tails[j].set_data(esttail_pos[:, 0], esttail_pos[:, 1])
                        est_n_tails[j].set_3d_properties(esttail_pos[:, 2])
                        artists.append(est_n_tails[j])

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

            ax.set_xlim(sxmin, sxmax)
            ax.set_ylim(symin, symax)
            ax.set_zlim(szmin, szmax)

        return artists

    num_frames = len(position)
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, interval=interval, blit=False
    )

    # Static limits if not following
    if not follow:
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

    if equal_aspect:
        ax.set_aspect('equal')
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

def draw_agent_circles(ax, positions, radius, color='red', linewidth=2):
    """
    Draws a horizontal circle (in the XY plane) of given radius around each agent.
    Args:
        ax: The matplotlib 3D axis.
        positions: (N, 3) array of agent positions.
        radius: Radius of the circle.
        color: Circle color.
        linewidth: Line width.
    Returns:
        List of Line3D objects (one per agent).
    """
    theta = np.linspace(0, 2 * np.pi, 100)
    lines = []
    for pos in positions:
        x = pos[0] + radius * np.cos(theta)
        y = pos[1] + radius * np.sin(theta)
        z = np.full_like(theta, pos[2])
        line, = ax.plot(x, y, z, color=color, linewidth=linewidth)
        lines.append(line)
    return lines

def _axes_limits(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    return (xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1])

def _clip_interval_to_limits(lo, hi, lim_lo, lim_hi):
    lo = lim_lo if not np.isfinite(lo) else lo
    hi = lim_hi if not np.isfinite(hi) else hi
    return max(lo, lim_lo), min(hi, lim_hi)

def _faces_for_wall_clipped_to_axes(wall, ax_limits):
    """
    Build faces so the wall shows only the portion inside current axis limits.
    Axis-aligned walls are clipped on tangent axes; thickness spans the normal axis.
    Non-axis-aligned walls use an oriented slab sized to axis span.
    """
    (xmin, xmax, ymin, ymax, zmin, zmax) = ax_limits
    n = np.asarray(getattr(wall, "normal", [1,0,0]), float)
    p = np.asarray(getattr(wall, "point", [0,0,0]), float)
    t = float(getattr(wall, "thickness", 0.1))
    bounds = getattr(wall, "bounds", None)

    aligned, axis_idx = _is_axis_aligned(n)
    if aligned:
        # set tangent bounds: use wall.bounds if present, then clip to axes
        if bounds is None:
            bx = (xmin, xmax)
            by = (ymin, ymax)
            bz = (zmin, zmax)
        else:
            bx, by, bz = bounds
            bx = _clip_interval_to_limits(bx[0], bx[1], xmin, xmax)
            by = _clip_interval_to_limits(by[0], by[1], ymin, ymax)
            bz = _clip_interval_to_limits(bz[0], bz[1], zmin, zmax)

        if axis_idx == 0:
            x0, x1 = p[0] - t*0.5, p[0] + t*0.5
            return _make_box_faces(x0, x1, by[0], by[1], bz[0], bz[1])
        elif axis_idx == 1:
            y0, y1 = p[1] - t*0.5, p[1] + t*0.5
            return _make_box_faces(bx[0], bx[1], y0, y1, bz[0], bz[1])
        else:  # axis_idx == 2
            z0, z1 = p[2] - t*0.5, p[2] + t*0.5
            return _make_box_faces(bx[0], bx[1], by[0], by[1], z0, z1)

    # Not axis-aligned: draw an oriented slab sized to the axis span (covers visible region)
    box_span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    size_xy = max(1e-6, 1.05 * box_span)  # pad slightly
    return _slab_faces_oriented(point=p, normal=n, thickness=t, size_xy=size_xy)

def _is_axis_aligned(n):
    n = np.asarray(n, float)
    i = int(np.argmax(np.abs(n)))
    axis = np.zeros(3)
    axis[i] = 1.0 if n[i] >= 0 else -1.0
    aligned = np.linalg.norm(n - axis) < 1e-6
    return aligned, i  # i in {0,1,2}

def _make_box_faces(xmin, xmax, ymin, ymax, zmin, zmax):
    # returns 6 quads (each as list of 3D points)
    return [
        # z-planes
        [[xmin,ymin,zmin],[xmax,ymin,zmin],[xmax,ymax,zmin],[xmin,ymax,zmin]],
        [[xmin,ymin,zmax],[xmax,ymin,zmax],[xmax,ymax,zmax],[xmin,ymax,zmax]],
        # y-planes
        [[xmin,ymin,zmin],[xmax,ymin,zmin],[xmax,ymin,zmax],[xmin,ymin,zmax]],
        [[xmin,ymax,zmin],[xmax,ymax,zmin],[xmax,ymax,zmax],[xmin,ymax,zmax]],
        # x-planes
        [[xmin,ymin,zmin],[xmin,ymax,zmin],[xmin,ymax,zmax],[xmin,ymin,zmax]],
        [[xmax,ymin,zmin],[xmax,ymax,zmin],[xmax,ymax,zmax],[xmax,ymin,zmax]],
    ]

def _orthonormal_basis_from_normal(n):
    n = np.asarray(n, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    # pick a ref not parallel to n
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    un = np.linalg.norm(u)
    if un < 1e-12:
        ref = np.array([0.0, 0.0, 1.0])
        u = np.cross(n, ref)
        un = np.linalg.norm(u)
    u /= (un + 1e-12)
    v = np.cross(n, u)
    return u, v, n

def _slab_faces_oriented(point, normal, thickness, size_xy):
    """
    Build faces (6 quads) for an oriented rectangular slab centered at 'point',
    with normal 'normal', finite thickness, and side length 'size_xy' in u and v.
    """
    u, v, n = _orthonormal_basis_from_normal(normal)
    h = thickness * 0.5
    s = size_xy * 0.5

    # two parallel rectangles (front/back faces)
    corners0 = [
        point + (-s)*u + (-s)*v - h*n,
        point + ( s)*u + (-s)*v - h*n,
        point + ( s)*u + ( s)*v - h*n,
        point + (-s)*u + ( s)*v - h*n,
    ]
    corners1 = [c + 2*h*n for c in corners0]

    # side faces connect corresponding edges
    faces = []
    faces.append(corners0)
    faces.append(corners1)
    for k in range(4):
        a0 = corners0[k]
        a1 = corners0[(k+1)%4]
        b0 = corners1[k]
        b1 = corners1[(k+1)%4]
        faces.append([a0, a1, b1, b0])
    return faces
