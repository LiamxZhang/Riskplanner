# 
import torch

# Extension APIs
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from envs.isaacgym_env import QuadrotorIsaacSim
from utils.grid_map import GridMap

class GroundTruthGridMap(GridMap):
    def __init__(self):
        """
        Initialize the GroundTruthGridMap class.
        """
        # Get the prim_grid entity as well as the dictionary
        prim_grid_class = QuadrotorIsaacSim().prim_grid
        self.prim_grid = prim_grid_class.prim_grid
        self.grid_resolution = prim_grid_class.grid_resolution

        # Extract all grid coordinates
        coords = torch.tensor(list(self.prim_grid.keys()), dtype=torch.float32)

        # Initialize the "GridMap" class
        super().__init__(
            grid_resolution=self.grid_resolution,
            coords = coords,
            dtype = torch.int32
        )

        self.prim_categories = {}  # Dictionary that mapping prims to IDs
        
        # construct the grid map
        for position, prim in self.prim_grid.items(): # position (Tuple), prim (USD.Prim)
            id = self._get_prim_id(prim)
            self._add_to_grid(id, position) # Add a prim to the grid map
        # print(f"Grid map: {self.grid_map}")
        # print(f"Prim categories: {self.prim_categories}")
        # self.visualize_scatter()
        
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
    
