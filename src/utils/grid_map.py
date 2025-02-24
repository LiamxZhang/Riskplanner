#!/usr/bin/env python
import torch
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from controller.backend import Backend
from configs.configs import MAP_ASSET
from ros2.point_publish_node import PointPublisher


class GridMap(Backend):
    """
    A class to create a basic grid map.
    """
    def __init__(self, grid_resolution, coords, dtype = torch.float32, map_name = "gridmap"):
        """
        Convert 3D real-map coordinates to grid map indices.
        Args:
            grid_resolution (float): The size of one map grid (square shape).
            coords (torch.tensor): A tensor containing all 3D point coordnates (at least the max and min points), 
                                    including the max and min of x,y,z
            dtype (torch.float32 or torch.int32): Data type of the map elememnt, default to be torch.float32
        """
        self.grid_resolution = grid_resolution
        self._compute_map_size(coords)
        self.dtype = dtype
        self.all_points = torch.empty((0, 3))
        self.points_max_length = MAP_ASSET["max_length_of_pointset"]

        # Initialize the grid map
        self.grid_map = torch.zeros(self.gridmap_size, dtype=dtype)

        if MAP_ASSET["ros2_publish"]:
            # Create MapPublisher node
            node_name = map_name + "_publish_node"
            topic_name = "/" + map_name + "/points"
            self.map_publisher = PointPublisher(node_name, topic_name)

    def _compute_map_size(self, coords):
        """
        Compute the map size based on the minimum and maximum values of coordinates.
        """
        if coords.numel() == 0:
            self.gridmap_size = [10, 10, 10]  # Default size if no prims exist
            self.realmap_bounds = torch.tensor([[0, 0, 0], [10.0, 10.0, 10.0]], dtype=torch.float32)
        else:
            # Real map size
            min_bounds = coords.min(dim=0).values - self.grid_resolution * MAP_ASSET["extend_units"]
            max_bounds = coords.max(dim=0).values + self.grid_resolution * (MAP_ASSET["extend_units"]+1)
            self.realmap_bounds = torch.stack([min_bounds, max_bounds]) # Shape: [2, 3]
            self.realmap_size = (max_bounds - min_bounds).tolist() # Type: list
            # print(f"Realmap bounds: {self.realmap_bounds}")
            # print(f"Realmap size: {self.realmap_size}")

            # Grid map size
            grid_min_bounds = torch.floor(min_bounds / self.grid_resolution)
            grid_max_bounds = torch.ceil(max_bounds / self.grid_resolution)
            self.gridmap_size = (grid_max_bounds - grid_min_bounds).to(torch.int32).tolist() # Type: list
            self.gridmap_bounds = torch.tensor([[0, 0, 0], self.gridmap_size], dtype=torch.float32)  # Shape: [2, 3]
            # print(f"Gridmap bounds: {self.gridmap_bounds}")
            # print(f"Gridmap size: {self.gridmap_size}")

            self.realmap_bounds_grided = torch.stack([grid_min_bounds, grid_max_bounds])  # Shape: [2, 3]
    
    def _add_to_grid(self, element, position):
        """
        Add a prim to the grid map at a specific position.
        Args:
            element (float or int): The number reprensenting the model or object.
            position (tuple): The 3D real position of the element.
        """
        if isinstance(position, torch.Tensor):
            position = position.clone().detach()
        else:
            position = torch.tensor(position, dtype=torch.float32)

        # Convert position to grid coordinates
        grid_coords = self.realmap_to_gridmap_index(position) # tuple

        if all(0 <= grid_coords[i] < self.gridmap_size[i] for i in range(len(self.gridmap_size))):
            self.grid_map[grid_coords] = element  # Store the element in the grid map

            position_tensor = position.unsqueeze(0)
            self.all_points = torch.cat((self.all_points, position_tensor), dim=0)
            if self.all_points.size(0) > self.points_max_length:
                self.all_points = self.all_points[-self.points_max_length:]
            return True
        else:
            print(f"Position {position} is out of bounds.")
            return False

    def realmap_to_gridmap_index(self, point):
        """
        Convert 3D real-map coordinates to grid map indices.
        Args:
            point (tensor or tuple): A tuple of shape (3) containing 3D points.
        Returns:
            indices (tuple): A tuple of shape (3) containing grid indices for each point.
        """
        if isinstance(point, torch.Tensor):
            processed_point = point.clone().detach()
        else:
            processed_point = torch.tensor(point, dtype=torch.float32)
        grid_index = torch.floor((processed_point - self.realmap_bounds[0]) / self.grid_resolution).to(torch.int32)
        return tuple(grid_index)
    
    def gridmap_to_realmap_index(self, grid_index):
        """
        Convert grid indices back to real-world coordinates
        Args:
            indices (tensor or tuple): A tuple of shape (3) containing grid indices (torch.int32) for each 3D point.
        Returns:
            real_pos (tuple): A tuple of shape (3) containing 3D points.
        """
        if isinstance(grid_index, torch.Tensor):
            processed_index = grid_index.clone().detach()
        else:
            processed_index = torch.tensor(grid_index, dtype=torch.float32)
        real_pos = processed_index * self.grid_resolution + self.realmap_bounds[0]
        return tuple(real_pos)
    
    def pub(self):
        """
        Publish all map points to ROS2 topic
        """
        if MAP_ASSET["ros2_publish"]:
            # self.map_publisher.pub(self.all_points)
            non_zero_grids = torch.nonzero(self.grid_map)
            self.map_publisher.pub(non_zero_grids.to(dtype=torch.float32))

    def get_local_map(self, center, local_map_shape):
        """
        Extract a local map from a 3D tensor, filling out-of-bound regions with a specified value.
        Parameters:
            center (torch.Tensor): The center of the local map (x, y, z) in realmap.
            local_map_shape (tuple): The size of the local map (x_size, y_size, z_size).
        Returns:
            local_map (torch.Tensor): The extracted local map with shape equal to `size`.
        """
        gridmap_size = torch.tensor(self.gridmap_size, dtype=torch.int32)
        if gridmap_size is None:
            gridmap_size = torch.tensor(self.grid_map.shape, dtype=torch.int32)

        # Calculate the start and end indices for the local map in global map
        center = torch.tensor(self.realmap_to_gridmap_index(center), dtype=torch.int32)
        size = torch.tensor(local_map_shape, dtype=torch.int32)
        start = center - size // 2
        end = center + size // 2 # + 1

        # Create the local map and fill it with the specified value
        local_map = torch.full(local_map_shape, MAP_ASSET["max_fill_value"], dtype=self.grid_map.dtype)

        # Determine the valid range within the global grid map
        valid_start = torch.maximum(start, torch.zeros_like(start))
        valid_end = torch.minimum(end, gridmap_size)

        # Determine the corresponding range within the local map
        local_start = torch.maximum(-start, torch.zeros_like(start))
        local_end = size - torch.maximum(end - gridmap_size, torch.zeros_like(end))

        # Copy the valid region from the grid map to the local map
        local_map[
            local_start[0]:local_end[0],
            local_start[1]:local_end[1],
            local_start[2]:local_end[2],
        ] = self.grid_map[
            valid_start[0]:valid_end[0],
            valid_start[1]:valid_end[1],
            valid_start[2]:valid_end[2],
        ]

        return local_map # .unsqueeze(0) 

    def get_local_map_points(self):
        """Convert grid map to point cloud coordinates"""
        occupied_indices = np.argwhere(self.grid_map.cpu().numpy() > 0)
        points = np.zeros((len(occupied_indices), 3), dtype=np.float32)
        
        # Convert grid indices to world coordinates
        points[:, 0] = occupied_indices[:, 2] * self.grid_resolution + self.realmap_bounds[0, 0]
        points[:, 1] = occupied_indices[:, 1] * self.grid_resolution + self.realmap_bounds[0, 1]
        points[:, 2] = occupied_indices[:, 0] * self.grid_resolution + self.realmap_bounds[0, 2]
        
        return points

    def visualize_grid(self):
        """
        Visualize the 3D grid map with a scatter plot.
        """
        non_empty_cells = torch.nonzero(self.grid_map)
        x, y, z = non_empty_cells[:, 0].cpu().numpy(), non_empty_cells[:, 1].cpu().numpy(), non_empty_cells[:, 2].cpu().numpy()
        values = self.grid_map[x, y, z].cpu().numpy()
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=values, cmap='viridis', s=100, alpha=0.8)
        plt.colorbar(sc, ax=ax, label='Prim ID')
        ax.set_title("3D Grid Map Visualization")
        ax.set_xlabel("X (grid)")
        ax.set_ylabel("Y (grid)")
        ax.set_zlabel("Z (grid)")
        plt.show()

    def visualize_scatter(self):
        """
        Visualize the 3D grid map with a scatter plot using real-world coordinates.
        """
        # Find non-empty grid cells
        non_empty_cells = torch.nonzero(self.grid_map)

        # create the figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # if all values of grid_map are 0，generate the virtual zero points
        if non_empty_cells.numel() == 0:
            x_min, y_min, z_min = self.realmap_bounds[0]
            x_max, y_max, z_max = self.realmap_bounds[1]
            # Set the axis range
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
        else:
            # Convert grid indices to real-world coordinates
            real_coords = torch.stack([
                torch.tensor(self.gridmap_to_realmap_index(grid_index), dtype=torch.float32) 
                for grid_index in non_empty_cells])
            x, y, z = real_coords[:, 0].cpu().numpy(), real_coords[:, 1].cpu().numpy(), real_coords[:, 2].cpu().numpy()
            values = self.grid_map[non_empty_cells[:, 0], non_empty_cells[:, 1], non_empty_cells[:, 2]].cpu().numpy()
            # Plot the scatter plot with real-world coordinates
            sc = ax.scatter(x, y, z, c=values, cmap='viridis', s=100, alpha=0.8)
            plt.colorbar(sc, ax=ax, label='Prim ID')

        ax.set_title("3D Grid Map Visualization (Real-world coordinates)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        plt.show()

    def update_state(self, state):
        """Method that when implemented, should handle the receival of the state of the vehicle using this callback

        Args:
            state (State): The current state of the vehicle.
        """
        self.p = state.position
        self.R = state.R
        self.orient = state.orient # as_euler('ZYX', degrees=True)
        self.v = state.linear_velocity
        self.w = state.angular_velocity

    def input_reference(self):
        """Method that when implemented, should return a list of desired angular velocities to apply to the vehicle rotors
        """
        return []

    def update(self, dt: float):
        """Method that when implemented, should be used to update the state of the backend and the information being sent/received
        from the communication interface. This method will be called by the simulation on every physics step

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """
        pass
    
    def start(self):
        """Method that when implemented should handle the begining of the simulation of vehicle
        """
        pass

    def stop(self):
        """Method that when implemented should handle the stopping of the simulation of vehicle
        """
        pass
    
    def reset(self):
        """Method that when implemented, should handle the reset of the vehicle simulation to its original state
        """
        pass


if __name__ == "__main__":
    coords = torch.tensor([
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0]
            ], dtype=torch.float32)
    
    # coords = torch.tensor([], dtype=torch.float32).reshape(0, 3)

    settings = {"grid_resolution": 1.0, "coords":coords, "dtype": torch.int32}
    gridmap = GridMap(**settings)
    gridmap.visualize_scatter()