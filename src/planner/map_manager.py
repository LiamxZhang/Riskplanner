# 
import torch
import carb

# Extension APIs
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from envs.isaacgym_env import QuadrotorIsaacSim


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
        
        self.ground_truth_grid_map()
        self.visualize_3d_grid()

    def ground_truth_grid_map(self):
        """
        Initialize the MapManager class.
        """
        self._compute_map_size()

        self.grid_map = torch.zeros(self.gridmap_size, dtype=torch.int)
        self.prim_categories = {}  # Dictionary that mapping prims to IDs

        # construct the grid map
        for position, prim in self.prim_grid.items():
            self._add_prim_to_grid(prim, position)
        # print(f"Grid map: {self.grid_map}")
        print(f"Prim categories: {self.prim_categories}")

    def _compute_map_size(self):
        """
        Compute the map size based on the minimum and maximum values of prim_grid.
        """
        if not self.prim_grid:
            self.realmap_size = [10.0, 10.0, 10.0]  # Default size if no prims exist
            self.map_bounds = [[0, 0, 0], [10.0, 10.0, 10.0]]
        else:
            # Extract all grid coordinates
            coords = torch.tensor(list(self.prim_grid.keys()), dtype=torch.float32)

            # Real map size
            min_bounds = coords.min(dim=0).values 
            max_bounds = coords.max(dim=0).values + 1
            self.realmap_bounds = [min_bounds.tolist(), max_bounds.tolist()]
            self.realmap_size = (max_bounds - min_bounds).tolist() 
            print(f"Realmap bounds: {self.realmap_bounds}")
            print(f"Realmap size: {self.realmap_size}")

            # Grid map size
            self.gridmap_bounds = [(min_bounds/self.grid_resolution).tolist(), 
                                   (max_bounds/self.grid_resolution).tolist()]
            self.gridmap_size = (
                                    int(self.realmap_size[0] / self.grid_resolution), 
                                    int(self.realmap_size[1] / self.grid_resolution), 
                                    int(self.realmap_size[2] / self.grid_resolution)
                                ) 
            print(f"Gridmap bounds: {self.gridmap_bounds}")
            print(f"Gridmap size: {self.gridmap_size}")

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

    def _add_prim_to_grid(self, prim, position):
        """
        Add a prim to the grid map at a specific position.
        Args:
            prim (Usd.Prim): The USD prim object.
            position (tuple): The 3D position of the prim.
        """
        # Convert position to grid coordinates
        position_tensor = torch.tensor(position, dtype=torch.float32)
        grid_coords = tuple((position_tensor / self.grid_resolution).long().tolist())
        if all(self.gridmap_bounds[0][i] <= grid_coords[i] < self.gridmap_bounds[1][i] for i in range(len(self.gridmap_size))):
            self.grid_map[grid_coords] = self._get_prim_id(prim)  # Store the ID in the grid map
        else:
            print(f"Position {position} is out of bounds.")

    def visualize_3d_grid(self):
        """
        Visualize the 3D grid map with a scatter plot.
        """
        import matplotlib.pyplot as plt
        
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

# Example usage
if __name__ == "__main__":
    # Initialize a risk map with grid size (100, 100, 50) and resolution 0.5
    risk_map = MapManager(grid_size=(100, 100, 50), grid_resolution=0.5)

    # Simulate a LiDAR point cloud
    point_cloud = torch.tensor([[1.2, 3.4, 0.5], [2.1, 3.9, 0.7], [1.1, 3.6, 0.4]], dtype=torch.float32)

    # Update the risk map with the point cloud
    risk_map.update_with_point_cloud(point_cloud)

    # Get the risk value at a specific position
    position = torch.tensor([1.0, 3.5, 0.5], dtype=torch.float32)
    risk_value = risk_map.get_risk_at_position(position)
    print(f"Risk value at position {position.tolist()}: {risk_value}")

    # Save the risk map to a file
    risk_map.save_map_to_file("risk_map.pt")

    # Clear the risk map
    risk_map.clear_risk_map()

    # Load the risk map from a file
    risk_map.load_map_from_file("risk_map.pt")
    print("Loaded risk map successfully.")
