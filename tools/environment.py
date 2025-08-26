"""
Generic base class for the simulation environment.
"""

import numpy as np

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