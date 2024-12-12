# This script setup the Obstacle Grid Map class

import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from utils.task_util import print_prim_and_grid


class ObstacleGrid:
    def __init__(self,gr):
        """
        Initialize the ObstacleGrid class
        gr (float): The grid resolution of the map
        """
        # prim_grid is a dictionary where the key is a 3D position, and the value is the prim name
        self._prim_grid = {}
        self.grid_resolution = gr
    
    @property
    def prim_grid(self):
        """
        Get the prim_grid property
        Return: A dictionary storing the obstacle in grid map
        """
        return self._prim_grid

    @prim_grid.setter
    def prim_grid(self, value):
        """
        Set the prim_grid property
        Args: 
            value: Must be a dictionary where keys are 3D position tuples and values are prim names
        """
        from pxr import Usd, Sdf

        if not isinstance(value, dict):
            raise ValueError("prim_grid must be a dictionary.")
        for key, prim in value.items():
            if not (isinstance(key, tuple) and len(key) == 3 and all(isinstance(coord, (int, float)) for coord in key)):
                raise ValueError("Keys in prim_grid must be 3D position tuples.")
            if not isinstance(prim, Usd.Prim):
                raise ValueError("Values in prim_grid must be Usd.Prim.")
        self._prim_grid = value


    def add(self, position, prim_entity):
        """
        Add a prim to a specific position
        Args:
            position: A 3D position tuple
            prim_entity: The prim model
        """
        if not (isinstance(position, tuple) and len(position) == 3 and all(isinstance(coord, (int, float)) for coord in position)):
            raise ValueError("Position must be a 3D tuple (x, y, z).")
        if position not in self._prim_grid:
            self._prim_grid[position] = prim_entity

    def remove_prim(self, position):
        """
        Remove a prim from a specific position
        Args:
            position: A 3D position tuple
        """
        if position in self._prim_grid:
            del self._prim_grid[position]
        else:
            raise KeyError(f"No prim found at position {position}.")

    def get_prim_at(self, point):
        """
        Get the prim path name at a map grid around input position point
        Args:
            position: A 3D position tuple
        Return: 
            The name of the prim path or None if not found
        """
        gr = self.grid_resolution
        grid_coord = (
            round(point[0] / gr) * gr,
            round(point[1] / gr) * gr,
            round(point[2] / gr) * gr
        )
        # print("grid_coord: ", grid_coord)

        # See whether the grid_coord exists in self._prim_grid
        if grid_coord in self._prim_grid:
            return self._prim_grid[grid_coord]  
        return None

    def show(self):
        print_prim_and_grid(self._prim_grid)


    def clear(self):
        """
        Clear all prim data
        """
        self._prim_grid.clear()

    def __repr__(self):
        """
        Return the string representation of the class
        """
        return f"ObstacleGridMap(prim_grid={self._prim_grid})"