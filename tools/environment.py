"""
Generic base class for the simulation environment.
"""

import numpy as np

_ALIGN_EPS = 1e-3

class Environment():
    def __init__(self):
        # Initialize parameters
        self.walls = []
        self.obstacles = []
        self.tunnels = []

    def add_wall(self, normal, point, thickness=0.1, bounds=None):
        """
        Add a wall to the environment.
        """
        wall = Wall(normal, point, thickness, bounds)
        self.walls.append(wall)
        return wall
    
    def add_tunnel(self, start, end, radius):
        """
        Add a tunnel to the environment.
        """
        tunnel = Tunnel(start, end, radius)
        self.tunnels.append(tunnel)
        return tunnel

    def distance_to_nearest_wall(self, pos, return_wall=False, return_vector=False):
        """
        Distance from 'pos' (3,) to the nearest wall surface across all walls.
        - If inside a wall slab, distance is the minimal distance to EXIT (>=0).
        - If no walls, returns np.inf (and None for vec/wall if requested).
        - Axis-aligned bounded walls: exact distance/vector to finite slab (AABB).
        - Non-axis-aligned bounded walls: falls back to infinite-slab approximation.
        - Tunnels are ignored (distance to wall material only).

        Parameters
        ----------
        pos : array-like (3,)
        return_wall : bool
            Also return the Wall object that yielded the minimum.
        return_vector : bool
            Also return the vector from pos to the closest point on the nearest wall.

        Returns
        -------
        If return_vector=False and return_wall=False:
            d
        If return_vector=False and return_wall=True:
            d, wall
        If return_vector=True and return_wall=False:
            d, vec
        If return_vector=True and return_wall=True:
            d, vec, wall
        """
        p = np.asarray(pos, dtype=float)
        if not self.walls:
            if return_vector and return_wall:
                return np.inf, None, None
            if return_vector:
                return np.inf, None
            if return_wall:
                return np.inf, None
            return np.inf

        best_d = np.inf
        best_vec = None
        best_w = None

        for w in self.walls:
            if w.bounds is None:
                vec = _vector_to_infinite_slab(w, p)
                d = float(np.linalg.norm(vec))
            else:
                aabb = _aabb_for_axis_aligned_wall(w)
                if aabb is not None:
                    vec = _vector_point_to_aabb(p, aabb)
                    d = float(np.linalg.norm(vec))
                else:
                    # bounded but not axis-aligned: approximate by infinite slab
                    vec = _vector_to_infinite_slab(w, p)
                    d = float(np.linalg.norm(vec))

            if d < best_d:
                best_d, best_vec, best_w = d, vec, w

        if return_vector and return_wall:
            return best_d, best_vec, best_w
        if return_vector:
            return best_d, best_vec
        if return_wall:
            return best_d, best_w
        return best_d
    
class Wall:
    def __init__(self, normal, point, thickness, bounds):
        """
        Wall defined as a slab of finite thickness.

        Parameters:
        - normal: np.array(3,) unit vector (direction of the wall's outward normal)
        - point:  np.array(3,) a reference point on the mid-plane
        - thickness: float, thickness of the wall
        - bounds: optional ((xmin, xmax), (ymin, ymax), (zmin, zmax)) for finite wall

        Usage example: env.add_wall(normal=[1,0,0], point=[0,0,0], thickness=5.0, bounds=((-10,10),(-10,10),(-1,1)))
        """
        self.normal = np.array(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)
        self.point = np.array(point, dtype=float)
        self.thickness = float(thickness)
        self.bounds = bounds

class Tunnel:
    def __init__(self, start, end, radius):
        """
        Cylindrical tunnel allowing agents to pass through a wall.

        Parameters:
        - start : np.array(3,)  -> one endpoint of the tunnel axis
        - end   : np.array(3,)  -> other endpoint of the tunnel axis
        - radius: float         -> tunnel radius

        Usage example: env.add_tunnel(start=[0,0,0], end=[5,0,0], radius=2.0)
        """
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)
        self.radius = float(radius)

        # Precompute direction + length
        self.axis = self.end - self.start
        self.length = np.linalg.norm(self.axis)
        if self.length == 0:
            raise ValueError("Tunnel start and end cannot be the same point.")
        self.axis_dir = self.axis / self.length

def _closest_distance_to_infinite_slab(wall, pos):
    """
    Distance to the nearest face of a finite-thickness (but infinite-extent) slab.
    If inside the slab, returns minimal distance to exit (>=0).
    """
    p = np.asarray(pos, dtype=float)
    s = float(np.dot(p - wall.point, wall.normal))  # signed dist to mid-plane
    half_t = 0.5 * wall.thickness
    if abs(s) <= half_t:
        return half_t - abs(s)
    return abs(s) - half_t

def _is_axis_aligned(normal):
    """
    (is_aligned, axis_index) where axis_index ∈ {0:x,1:y,2:z}.
    """
    n = np.asarray(normal, dtype=float)
    idx = int(np.argmax(np.abs(n)))
    axis = np.zeros(3); axis[idx] = 1.0 if n[idx] >= 0 else -1.0
    aligned = np.linalg.norm(n - axis) < _ALIGN_EPS
    return aligned, idx

def _aabb_for_axis_aligned_wall(wall):
    """
    Build an AABB for an axis-aligned finite wall slab (global axis bounds + thickness).
    Returns ((xmin,xmax),(ymin,ymax),(zmin,zmax)) or None if not axis-aligned or no bounds.
    """
    if wall.bounds is None:
        return None
    aligned, axis = _is_axis_aligned(wall.normal)
    if not aligned:
        return None

    half_t = 0.5 * wall.thickness
    px, py, pz = wall.point
    (bxmin, bxmax), (bymin, bymax), (bzmin, bzmax) = wall.bounds

    if axis == 0:     # normal ~ ±x -> thickness spans x
        return ((px - half_t, px + half_t), (bymin, bymax), (bzmin, bzmax))
    elif axis == 1:   # normal ~ ±y -> thickness spans y
        return ((bxmin, bxmax), (py - half_t, py + half_t), (bzmin, bzmax))
    else:             # axis == 2: normal ~ ±z -> thickness spans z
        return ((bxmin, bxmax), (bymin, bymax), (pz - half_t, pz + half_t))

def _distance_point_to_aabb(pos, aabb):
    """
    Euclidean distance from a point to an AABB. If inside, min distance to any face.
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = aabb
    x, y, z = np.asarray(pos, dtype=float)
    dx = 0.0 if xmin <= x <= xmax else min(abs(x - xmin), abs(x - xmax))
    dy = 0.0 if ymin <= y <= ymax else min(abs(y - ymin), abs(y - ymax))
    dz = 0.0 if zmin <= z <= zmax else min(abs(z - zmin), abs(z - zmax))
    if dx > 0 or dy > 0 or dz > 0:
        return float(np.sqrt(dx*dx + dy*dy + dz*dz))
    return float(min(x - xmin, xmax - x, y - ymin, ymax - y, z - zmin, zmax - z))

def _vector_to_infinite_slab(wall, pos):
    """
    Vector from pos to the nearest point on the infinite-extent slab of thickness wall.thickness.
    (Add this vector to pos to land exactly on the closest face.)
    """
    p = np.asarray(pos, dtype=float)
    n = wall.normal
    s = float(np.dot(p - wall.point, n))     # signed dist to mid-plane
    half_t = 0.5 * wall.thickness

    if abs(s) > half_t:
        # outside: go back to the nearest face along -sign(s) * n
        d = abs(s) - half_t
        return -np.sign(s) * d * n
    else:
        # inside: go to the nearer face (+n if s>=0, else -n)
        d = (half_t - abs(s))
        dir_n = n if s >= 0.0 else -n
        return d * dir_n

def _vector_point_to_aabb(pos, aabb):
    """
    Vector from pos to the closest point on (or in) the AABB surface.
    If pos is outside the box, this is the straight-line vector to the box.
    If pos is inside, it points to the nearest face (shortest exit).
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = aabb
    x, y, z = np.asarray(pos, dtype=float)

    inside_x = xmin <= x <= xmax
    inside_y = ymin <= y <= ymax
    inside_z = zmin <= z <= zmax

    # If outside in any axis, clamp to the box on that axis
    cx = min(max(x, xmin), xmax)
    cy = min(max(y, ymin), ymax)
    cz = min(max(z, zmin), zmax)

    if not (inside_x and inside_y and inside_z):
        # Outside: closest point is (cx, cy, cz)
        return np.array([cx - x, cy - y, cz - z], dtype=float)

    # Inside: go to nearest face (smallest displacement)
    dx = min(x - xmin, xmax - x)
    dy = min(y - ymin, ymax - y)
    dz = min(z - zmin, zmax - z)

    if dx <= dy and dx <= dz:
        # move along x to nearest face
        if (x - xmin) <= (xmax - x):
            return np.array([xmin - x, 0.0, 0.0], dtype=float)
        else:
            return np.array([xmax - x, 0.0, 0.0], dtype=float)
    elif dy <= dz:
        if (y - ymin) <= (ymax - y):
            return np.array([0.0, ymin - y, 0.0], dtype=float)
        else:
            return np.array([0.0, ymax - y, 0.0], dtype=float)
    else:
        if (z - zmin) <= (zmax - z):
            return np.array([0.0, 0.0, zmin - z], dtype=float)
        else:
            return np.array([0.0, 0.0, zmax - z], dtype=float)