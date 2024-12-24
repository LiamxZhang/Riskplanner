# 
import torch

# Extension APIs
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from envs.isaacgym_env import QuadrotorIsaacSim
from map.groundtruth_gridmap import GroundTruthGridMap
from map.sense_gridmap import SenseGridMap


class MapManager:
    """
    A class to manage a risk map based on LiDAR point cloud data.
    """
    def __init__(self):
        """
        Initialize the MapManager class.
        """
        # self.risk_map = torch.zeros(grid_size, dtype=torch.float32)  # Initialize risk values to zero
        # Get the 
        prim_grid_class = QuadrotorIsaacSim().prim_grid
        self.prim_grid = prim_grid_class.prim_grid
        self.grid_resolution = prim_grid_class.grid_resolution
        
        self.gt_gridmap = GroundTruthGridMap()

    def update_with_point_cloud(self, point_cloud):
        """
        Update the risk map based on LiDAR point cloud data.

        Args:
            point_cloud (torch.Tensor): A tensor of shape (N, 3) representing LiDAR points.
        """
        if not isinstance(point_cloud, torch.Tensor) or point_cloud.shape[1] != 3:
            raise ValueError("Point cloud must be a torch.Tensor of shape (N, 3).")

        # Convert point cloud coordinates to grid indices
        indices = torch.div(point_cloud, self.grid_resolution, rounding_mode='floor').long()

        # Clamp indices to be within the grid bounds
        for i, size in enumerate(self.grid_size):
            indices[:, i] = torch.clamp(indices[:, i], min=0, max=size - 1)

        # Increment risk values in corresponding grid cells
        for idx in indices:
            self.risk_map[idx[0], idx[1], idx[2]] += 1.0

    def get_risk_at_position(self, position):
        """
        Get the risk value at a specific position.

        Args:
            position (torch.Tensor): A tensor of shape (3,) representing a 3D position.

        Returns:
            float: The risk value at the specified position.
        """
        if not isinstance(position, torch.Tensor) or position.shape != (3,):
            raise ValueError("Position must be a torch.Tensor of shape (3,).")

        # Convert position to grid indices
        indices = torch.div(position, self.grid_resolution, rounding_mode='floor').long()

        # Clamp indices to be within the grid bounds
        indices = torch.clamp(indices, min=0, max=torch.tensor(self.grid_size) - 1)

        return self.risk_map[indices[0], indices[1], indices[2]].item()

    def clear_risk_map(self):
        """
        Reset all risk values in the map to zero.
        """
        self.risk_map.zero_()

    def save_map_to_file(self, filename):
        """
        Save the risk map to a file.

        Args:
            filename (str): The file name to save the risk map.
        """
        torch.save(self.risk_map, filename)

    def load_map_from_file(self, filename):
        """
        Load a risk map from a file.

        Args:
            filename (str): The file name to load the risk map from.
        """
        self.risk_map = torch.load(filename)
