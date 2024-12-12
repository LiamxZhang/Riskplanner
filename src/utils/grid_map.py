import torch
import matplotlib.pyplot as plt

class GridMap:
    """
    A class to create a basic grid map.
    """
    def __init__(self, grid_resolution, coords, dtype = torch.float32):
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

        # Initialize the grid map
        self.grid_map = torch.zeros(self.gridmap_size, dtype=dtype)

    def _compute_map_size(self, coords):
        """
        Compute the map size based on the minimum and maximum values of coordinates.
        """
        if coords.numel() == 0:
            self.gridmap_size = [10, 10, 10]  # Default size if no prims exist
            self.realmap_bounds = torch.tensor([[0, 0, 0], [10.0, 10.0, 10.0]], dtype=torch.float32)
        else:
            # Real map size
            min_bounds = coords.min(dim=0).values 
            max_bounds = coords.max(dim=0).values + self.grid_resolution
            self.realmap_bounds = torch.stack([min_bounds, max_bounds]) # Shape: [2, 3]
            self.realmap_size = (max_bounds - min_bounds).tolist() 
            # print(f"Realmap bounds: {self.realmap_bounds}")
            # print(f"Realmap size: {self.realmap_size}")

            # Grid map size
            grid_min_bounds = torch.floor(min_bounds / self.grid_resolution)
            grid_max_bounds = torch.ceil(max_bounds / self.grid_resolution)
            self.gridmap_bounds = torch.stack([grid_min_bounds, grid_max_bounds])  # Shape: [2, 3]
            self.gridmap_size = (grid_max_bounds - grid_min_bounds).to(torch.int32).tolist() 
            # print(f"Gridmap bounds: {self.gridmap_bounds}")
            # print(f"Gridmap size: {self.gridmap_size}")
    
    def _add_to_grid(self, element, position):
        """
        Add a prim to the grid map at a specific position.
        Args:
            element (float or int): The number reprensenting the model or object.
            position (tuple): The 3D real position of the element.
        """
        # Convert position to grid coordinates
        grid_coords = self.realmap_to_gridmap_index(position) # tuple

        if all(0 <= grid_coords[i] < self.gridmap_size[i] for i in range(len(self.gridmap_size))):
            self.grid_map[grid_coords] = element  # Store the element in the grid map
        else:
            print(f"Position {position} is out of bounds.")

    def realmap_to_gridmap_index(self, point):
        """
        Convert 3D real-map coordinates to grid map indices.
        Args:
            point (tensor or tuple): A tuple of shape (3) containing 3D points.
        Returns:
            indices (tuple): A tuple of shape (3) containing grid indices for each point.
        """
        grid_index = torch.floor((torch.tensor(point) - self.realmap_bounds[0]) / self.grid_resolution).to(torch.int32)
        return tuple(grid_index)
    
    def gridmap_to_realmap_index(self, grid_index):
        """
        Convert grid indices back to real-world coordinates
        Args:
            indices (tensor or tuple): A tuple of shape (3) containing grid indices (torch.int32) for each 3D point.
        Returns:
            real_pos (tuple): A tuple of shape (3) containing 3D points.
        """
        real_pos = torch.tensor(grid_index) * self.grid_resolution + self.realmap_bounds[0]
        return tuple(real_pos)
    
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