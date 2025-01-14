#!/usr/bin/env python
import torch

# Extension APIs
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from envs.isaacgym_env import QuadrotorIsaacSim
from utils.grid_map import GridMap


class SenseGridMap(GridMap):
    def __init__(self):
        """
        Initialize the GroundTruthGridMap class.
        """
        # Get the prim_grid entity as well as the dictionary
        self.prim_grid_class = QuadrotorIsaacSim().prim_grid
        self.prim_grid = self.prim_grid_class.prim_grid
        self.grid_resolution = self.prim_grid_class.grid_resolution

        # Extract all grid coordinates to get the map size
        coords = torch.tensor(list(self.prim_grid.keys()), dtype=torch.float32)

        # Initialize the "GridMap" class
        super().__init__(
            grid_resolution=self.grid_resolution,
            coords = coords,
            dtype = torch.int32
        )
        self.points_max_length = int(1e2)
        
        # Initialize the to-be-updated variables
        self.reset()
        self.grid_lifecycle = 100 # for sensed grids
        # print(f"Grid map: {self.grid_map}")
        # print(f"Prim categories: {self.prim_categories}")

    def start(self):
        """Method that when implemented should handle the begining of the simulation of vehicle
        """
        pass

    def _get_prim_id(self, prim):
        """
        Get a unique ID for a prim based on its path or name.
        Args:
            prim (Usd.Prim): The USD prim object.
        Returns:
            int: Unique ID assigned to this prim.
        """
        prim_identifier = prim.GetPath().pathString  # Use path as identifier
        if prim_identifier not in self.prim_categories:
            self.prim_categories[prim_identifier] = len(self.prim_categories) + 1
        return self.prim_categories[prim_identifier]
    
    def update(self, lidar_data):
        """
        Update the grid map based on LiDAR point cloud data.
        Args:
            lidar_data (torch.tensor): A tensor of shape (N, 3) containing LiDAR-detected 3D points.
        """
        for point in lidar_data:        
            # Convert real-world position to grid coordinates
            grid_index = self.realmap_to_gridmap_index(point)
            # print("grid_index: ", grid_index)
            if not all(0 <= grid_index[i] < self.gridmap_size[i] for i in range(len(grid_index))):
                continue  # Jump over the outlier points
            
            if self.grid_updated[grid_index].item() > 0:
                continue # Jump over the points that have been updated recently
            
            # Get the prim entity at the index
            prim = self.prim_grid_class.get_prim_at(point)
            if prim is None:
                continue # Jump over the points that corresponds to nothing
            
            # add to grid
            id = self._get_prim_id(prim)
            if self._add_to_grid(id, point):
                self.grid_updated[grid_index] = self.grid_lifecycle

        # Decrement all values in self.grid_updated by 1, with a minimum of 0
        self.grid_updated = torch.maximum(self.grid_updated - 1, torch.tensor(0, dtype=torch.int32))
        # print("self.grid_updated: ", self.grid_updated)

        # if not torch.all(self.grid_updated==0):
        #     self.visualize_scatter()
        # Publish points to ROS2
        self.pub()
    
    def reset(self):
        """Method that when implemented, should handle the reset of the vehicle simulation to its original state
        """
        # Reset the grid map
        self.grid_map = torch.zeros(self.gridmap_size, dtype=self.dtype)
        # All nonzero element will be deducted by 1 in every update
        self.grid_updated = torch.zeros(self.gridmap_size, dtype=torch.int32)
        # All prims will be classified as IDs
        self.prim_categories = {}  # Dictionary that mapping prims to IDs
        # All points collections will be cleared
        self.all_points = torch.empty((0, 3))


            