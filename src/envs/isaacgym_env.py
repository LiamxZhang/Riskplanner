#!/usr/bin/env python
# This script setup the isaac sim environment
#
import sys
import math
import time
import carb
from threading import Lock
import torch
import rclpy

try:
    import trimesh
except ImportError as e:
    import subprocess
    install_command = [sys.executable, "-m", "pip", "install", "trimesh", "rtree"]
    result = subprocess.run(install_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    import trimesh
    
from scipy.spatial import Delaunay

import omni
from omni.isaac.kit import SimulationApp

#from omni.isaac.occupancy_map import _occupancy_map
#import omni.isaac.core.utils.prims as prims_utils

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from utils.obstacle_grid import ObstacleGrid
from utils.task_util import is_masked
from configs.configs import APP_SETTINGS, MAP_ASSET, WORLD_SETTINGS, CONTROL_PARAMS

class QuadrotorIsaacSim:
    """
    QuadrotorIsaacSim is a singleton class (there is only one object instance at any given time) that will be used to 
    """

    # The object instance
    _instance = None
    _is_initialized = False

    # Lock for safe multi-threading
    _lock: Lock = Lock()

    def __init__(self):
        """
        Initialize the QuadrotorIsaacSim environment.
        Args:
            config: configuration parameters.
        """
        # If we already have an instance of the PegasusInterface, do not overwrite it!
        if QuadrotorIsaacSim._is_initialized:
            return

        QuadrotorIsaacSim._is_initialized = True
        
        # Create the app
        self.App = SimulationApp(launch_config=APP_SETTINGS)
        # usd_path = MAP_ASSET["NYC_usd_path"]
        # self.masked_prims = MAP_ASSET["masked_prims"]["NYC_usd_path"]
        usd_path=None
        self.load_task(usd_path)
        time.sleep(2)
        self.init_config(CONTROL_PARAMS['grid_resolution'],CONTROL_PARAMS['control_cycle'])

        # Access the world
        from omni.isaac.core.world import World
        self._world = World(**WORLD_SETTINGS)

        # Scan the map & save prims in grids
        self.start()

        # Activate the ROS2 kernel
        rclpy.init(args=None)

    """
    Properties
    """

    @property
    def world(self):
        """The current omni.isaac.core.world World instance

        Returns:
            omni.isaac.core.world: The world instance
        """
        return self._world

    @property
    def time(self):
        """The current omni.isaac.core SimulationContext time 

        Returns:
            SimulationContext.time : The current_time value
        """
        return self.simulation_context.current_time

    """
    Operations
    """

    def start(self):
        self.reset()
        self.save_all_prims_in_grid() 
        self.init_timer()

    def stop(self):
        self.App.close()  # Cleanup application
        sys.exit()

    def update(self):
        self.App.update()

    def reset(self):
        self._world.reset()

    def is_running(self):
        return self.App.is_running() and not self.App.is_exiting()

    def load_task(self, usd_path=None):
        """
        Loads a new task into the environment and resets task-related variables.
        Records the mask IDs of the robot, gripper, and objects in the scene.
        Args:
            usd_path (str): Path of the task object or the environment USD.
        """
        from omni.isaac.core.utils.nucleus import get_assets_root_path, is_file
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            self.stop_task()
        # Set task directory
        if not usd_path: 
            usd_path = assets_root_path + MAP_ASSET["default_usd_path"]
            self.masked_prims = MAP_ASSET["masked_prims"]["default_usd_path"]

        # Load the stage
        if is_file(usd_path):
            omni.usd.get_context().open_stage(usd_path)
        else:
            carb.log_error(
                f"the usd path {usd_path} could not be opened, please make sure that {usd_path} is a valid usd file in {assets_root_path}"
            )
            self.stop_task()
        # Wait two frames so that stage starts loading
        self.App.update()
        self.App.update()
        print("Loading stage...")

        from omni.isaac.core.utils.stage import is_stage_loading
        while is_stage_loading():
            self.App.update()
        print("Loading Complete")
    
    def init_config(self, gr, dt=0.06):
        """
        Initialize the grid map resolution parameters and also the interface to access the convex meshes.
        Args:
            gr (float): grid resolution. 
            dt (float): time step during each simulation cycle. 
        """
        # for grid map
        self.stage = omni.usd.get_context().get_stage()
        self.length_unit = self.stage.GetMetadata('metersPerUnit')
        self.grid_resolution = gr / self.length_unit # number is in meter; less than 1.0 and can divide 1.0

        # 
        self.prim_grid = ObstacleGrid(self.grid_resolution)
        # Initiate the interface to access convex mesh data
        from omni.physx import get_physx_cooking_interface, get_physx_interface
        get_physx_interface().force_load_physics_from_usd()
        self.cooking_interface = get_physx_cooking_interface()
        
        # Initiate the timer parameters
        self.timestep_threshold = dt
        self.previous_simulation_time = 0.0
        
    def init_timer(self):
        from omni.isaac.core import SimulationContext
        self.simulation_context = SimulationContext()
        self.previous_simulation_time = self.simulation_context.current_time

    def get_timestep(self):
        """
        The actual simulation time step, according to user-defined time step (self.timestep_threshold) in init_config.
        Returns:
            timestep (float): actual time step. 
        """
        current_simulation_time = self.simulation_context.current_time
        timestep = current_simulation_time - self.previous_simulation_time

        if timestep >= self.timestep_threshold:
            self.previous_simulation_time = current_simulation_time
            return timestep
        
        return 0
    
    def is_collided_with_prim(self, prim, point, threshold=None):
        """
        Judge whether the point collides with the prim
        Args: 
            threshold: it is a collision, only when the point-prim distance < the threshold
        Returns the flag: whether point collides with the prim
        """
        from pxr import UsdGeom, Usd, Gf

        transform_matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        prim_path = str(prim.GetPath())
        # print("prim_path: ", prim_path)
        num_convex_hulls = self.cooking_interface.get_nb_convex_mesh_data(prim_path)
        
        if num_convex_hulls == 0:
            return False

        if threshold is None: # set the deault value
            threshold = 1e-4 / self.length_unit

        for hull_index in range(num_convex_hulls):
            convex_hull_data = self.cooking_interface.get_convex_mesh_data(prim_path, hull_index)
            # get vertices & polygons
            vertices = convex_hull_data["vertices"]
            
            vertex_world_list = []
            for vertex in vertices:
                vert = torch.tensor(vertex, dtype=torch.float32)
                vertex_world = transform_matrix.Transform(Gf.Vec3d(vert[0].item(), vert[1].item(), vert[2].item()))
                vertex_world_list.append([vertex_world[0], vertex_world[1], vertex_world[2]])
            
            # print(f"vertex_world_list: {vertex_world_list}")
            hull = Delaunay(vertex_world_list)
            mesh = trimesh.Trimesh(vertices=vertex_world_list, faces=hull.simplices)
            # distance = mesh.nearest.signed_distance([point]) # another slower method
            closest_point, distance, _ = trimesh.proximity.closest_point(mesh, [point])
            if distance<=threshold:
                # print(f"Distance: {distance}, between {point} and closest point: {closest_point}")
                return True
        return False

    def get_bounding_box(self, prim):
        """ return min and max coordinates of bounding box in centimeter unit"""
        from pxr import UsdGeom, Usd, Gf
        # Get BBoxCache, and calculate bounding box in local frame
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        box = bbox.ComputeWorldBound(prim).GetRange()

        # Get the local to world transform
        xformable = UsdGeom.Xformable(prim)
        transform_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        self.prim_position = transform_matrix.ExtractTranslation()

        # Get min and max bounding box 
        min_bbox_local = box.GetMin()
        max_bbox_local = box.GetMax()

        # To world frame
        min_bbox_world = transform_matrix.Transform(Gf.Vec3d(min_bbox_local))  # in centimeter unit
        max_bbox_world = transform_matrix.Transform(Gf.Vec3d(max_bbox_local)) 

        # To world frame
        min_bbox = Gf.Vec3d(
            min(min_bbox_world[0], max_bbox_world[0]),
            min(min_bbox_world[1], max_bbox_world[1]),
            min(min_bbox_world[2], max_bbox_world[2])
        )
        max_bbox = Gf.Vec3d(
            max(min_bbox_world[0], max_bbox_world[0]),
            max(min_bbox_world[1], max_bbox_world[1]),
            max(min_bbox_world[2], max_bbox_world[2])
        )
        return min_bbox, max_bbox

    def save_all_prims_in_grid(self):
        """
        Create a dictionary to save all prims in corresponding grids in the map.
        The dictionary key is map grid, and the value is the prim.
        Returns:
            self.prim_grid (dict): all prims in grid. 
        """
        # from pxr import Gf
        # stage = omni.usd.get_context().get_stage()

        gr = self.grid_resolution

        # get all prims from the stage
        for prim in self.stage.Traverse():
            prim_path = str(prim.GetPath())
            # check whether prim_path is masked
            if is_masked(prim_path, self.masked_prims):
                continue
            
            num_convex_hulls = self.cooking_interface.get_nb_convex_mesh_data(prim_path)
            
            if num_convex_hulls > 0:
                min_bbox, max_bbox = self.get_bounding_box(prim)
                x_range = torch.arange(math.floor(min_bbox[0] / gr) * gr, math.ceil(max_bbox[0] / gr) * gr + gr, gr).tolist()
                y_range = torch.arange(math.floor(min_bbox[1] / gr) * gr, math.ceil(max_bbox[1] / gr) * gr + gr, gr).tolist()
                z_range = torch.arange(math.floor(min_bbox[2] / gr) * gr, math.ceil(max_bbox[2] / gr) * gr + gr, gr).tolist()
                
                for x in x_range:
                    for y in y_range:
                        for z in z_range:
                            # point = torch.tensor([x, y, z], dtype=torch.float32)
                            if self.is_collided_with_prim(prim, [x, y, z], gr/2):
                                self.prim_grid.add((x, y, z), prim)
                # print("Prim: ", prim)
                # print("Prim position: ", self.prim_position)
                # print("Min bounding_box: ", min_bbox)
                # print("Max bounding_box: ", max_bbox)
        self.prim_grid.show()
        return self.prim_grid
            

    def __new__(cls):
        """Allocates the memory and creates the actual QuadrotorIsaacSim object is not instance exists yet. Otherwise,
        returns the existing instance of the QuadrotorIsaacSim class.

        Returns:
            cls: the single instance of the QuadrotorIsaacSim class
        """

        # Use a lock in here to make sure we do not have a race condition
        # when using multi-threading and creating the first instance of the QuadrotorIsaacSim
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(QuadrotorIsaacSim, cls).__new__(cls)

        return cls._instance

    def __del__(self):
        """Destructor for the object. Destroys the only existing instance of this class."""
        QuadrotorIsaacSim._instance = None
        QuadrotorIsaacSim._is_initialized = False

if __name__ == "__main__":
    # CONFIG = {"app":{"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"},
    #           "usd_path": None,
    #           "grid_resolution": 1.0,
    #           "control_cycle": 0.1 
    #           }
    QIS = QuadrotorIsaacSim()

    while QIS.is_running():
        # print("current simulation time: ", QIS.time)
        QIS.update()
    QIS.stop()
