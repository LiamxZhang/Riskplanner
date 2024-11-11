# This script setup the isaac sim environment
#
import os
import sys
import math
import time
import numpy as np
import omni
from omni.isaac.kit import SimulationApp
#from omni.isaac.occupancy_map import _occupancy_map
#import omni.isaac.core.utils.prims as prims_utils

sys.path.append('..')
from utils.task_util import to_world, point_to_plane_distance, print_prim_and_grid, is_masked
from envs.prim_class import PrimClass

class QuadrotorIsaacSim():
    """
    Initializes the QuadrotorIsaacSim environment.
    Args:
        visualizer: interface, optional.
    Attributes:
        _prev_arm_action (numpy.ndarray): Stores the previous arm action.
    """
    def __init__(self, config, usd_path=None):
        # 
        self.App = SimulationApp(launch_config=config['app'])
        self.load_task(usd_path)
        time.sleep(2)
        self.init_config(config['grid_resolution'])

    def load_task(self, usd_path=None):
        """
        Loads a new task into the environment and resets task-related variables.
        Records the mask IDs of the robot, gripper, and objects in the scene.
        Args:
            usd_path (str): Path of the task object or the environment USD.
        """
        import carb
        from omni.isaac.core.utils.nucleus import get_assets_root_path, is_file
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            self.stop_task()
        # Set task directory
        if not usd_path: 
            usd_path = assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd"

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
    
    def init_config(self, gr):
        self.masked_prims = ['Looks','Meshes','Lighting','Road','Buildings','Pavement',
                        'GroundPlane','TrafficLight','Bench','Tree','TableChair',
                        'Billboard','Lamp','RoadBarriers','Booth','Umbrella','Camera']
        # for grid map
        stage = omni.usd.get_context().get_stage()
        self.length_unit = stage.GetMetadata('metersPerUnit')
        self.grid_resolution = gr / self.length_unit # number is in meter; less than 1.0 and can divide 1.0

        # Initiate the interface to access convex mesh data
        from omni.physx import get_physx_cooking_interface, get_physx_interface
        get_physx_interface().force_load_physics_from_usd()
        self.cooking_interface = get_physx_cooking_interface()

    def init_timer(self, dt=0.1):
        from omni.isaac.core import SimulationContext
        self.simulation_context = SimulationContext()
        self.timestep_threshold = dt
        self.previous_simulation_time = self.simulation_context.current_time

    def get_timestep(self):
        current_simulation_time = self.simulation_context.current_time
        timestep = current_simulation_time - self.previous_simulation_time

        if timestep >= self.timestep_threshold:
            self.previous_simulation_time = current_simulation_time
            return timestep
        
        return 0

    def get_simulation_time(self):
        return self.simulation_context.current_time

    def stop_task(self):
        self.App.close()  # Cleanup application
        sys.exit()

    def update(self):
        self.App.update()

    def is_running(self):
        return self.App._app.is_running() and not self.App.is_exiting()

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

        # distances_flags = []
        for hull_index in range(num_convex_hulls):
            convex_hull_data = self.cooking_interface.get_convex_mesh_data(prim_path, hull_index)
            # get vertices & polygons
            vertices = convex_hull_data["vertices"]
            polygons = convex_hull_data["polygons"]
            
            # For each polygon of hull
            for poly_index in range(convex_hull_data["num_polygons"]):
                index_base = polygons[poly_index]["index_base"]
                
                # collect all vertices of polygon
                poly_world_vertex = []
                for vertex_index in range(polygons[poly_index]["num_vertices"]):
                    current_index = convex_hull_data["indices"][index_base + vertex_index]
                    # poly_world_vertex.append(to_world(vertices[current_index], transform_matrix))
                    vert =  np.fromiter(vertices[current_index], dtype=np.float64)
                    vertex_world = transform_matrix.Transform(Gf.Vec3d(vert[0],vert[1],vert[2]))
                    poly_world_vertex.append(vertex_world)
                # print("poly_world_vertex: ", poly_world_vertex)
                distance, is_inside_poly = point_to_plane_distance(point, poly_world_vertex)
                if distance<=threshold and is_inside_poly:
                    return True
                # distances_flags.append((distance,is_inside_poly))
                
        # min_distance, inside_poly = min(distances_flags, key=lambda x: x[0])
        # print("min_distance: ", min_distance, "inside_poly?: ", inside_poly)
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
        # from pxr import Gf
        # get current stage
        stage = omni.usd.get_context().get_stage()

        gr = self.grid_resolution
        self.prim_grid = {}
        
        # get all prims from the stage
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            # check whether prim_path is masked
            if is_masked(prim_path, self.masked_prims):
                continue

            num_convex_hulls = self.cooking_interface.get_nb_convex_mesh_data(prim_path)

            if num_convex_hulls > 0:
                min_bbox, max_bbox = self.get_bounding_box(prim)
                x_range = np.arange(math.floor(min_bbox[0] / gr) * gr, math.ceil(max_bbox[0] / gr) * gr + gr, gr).tolist()
                y_range = np.arange(math.floor(min_bbox[1] / gr) * gr, math.ceil(max_bbox[1] / gr) * gr + gr, gr).tolist()
                z_range = np.arange(math.floor(min_bbox[2] / gr) * gr, math.ceil(max_bbox[2] / gr) * gr + gr, gr).tolist()
                
                for x in x_range:
                    for y in y_range:
                        for z in z_range:
                            # print("Checking point: ", np.array([x, y, z]))                   
                            if self.is_collided_with_prim(prim, np.array([x, y, z]), gr/2):
                                point_key = (x, y, z)
                                if point_key not in self.prim_grid:
                                    self.prim_grid[point_key] = prim
                # print("Prim: ", prim)
                # print("Prim position: ", self.prim_position)
                # print("Min bounding_box: ", min_bbox)
                # print("Max bounding_box: ", max_bbox)
        print_prim_and_grid(self.prim_grid)
        return self.prim_grid

    def detect_prim_from_position(self, point):
        gr = self.grid_resolution
        grid_coord = (
            round(point[0] / gr) * gr,
            round(point[1] / gr) * gr,
            round(point[2] / gr) * gr
        )
        # print("grid_coord: ", grid_coord)
        if grid_coord in self.prim_grid:
            return self.prim_grid[grid_coord]  # 返回对应的 prim
        return None

    def show_grid_by_prim(self, prim_grid):
        from collections import defaultdict
        grouped_grid = defaultdict(list)

        # put grid_center of same prim into list
        for grid_center, prim in prim_grid.items():
            grouped_grid[prim].append(grid_center)

        for prim, grid_centers in grouped_grid.items():
            # prim_path = str(prim.GetPath())
            # if "Floor" in prim_path:
            print(f"Prim: {prim}")
            print("Grid centers: ", grid_centers)
            
    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """
        Retrieves the entire scene's 3D point cloud observations and colors.

        Args:
            ignore_robot (bool): Whether to ignore points corresponding to the robot.
            ignore_grasped_obj (bool): Whether to ignore points corresponding to grasped objects.

        Returns:
            tuple: A tuple containing scene points and colors.
        """
        pass

    def get_3d_obs_by_name(self, query_name):
        """
        Retrieves 3D point cloud observations and normals of an object by its name.

        Args:
            query_name (str): The name of the object to query.

        Returns:
            tuple: A tuple containing object points and object normals.
        """
        pass

if __name__ == "__main__":
    CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}
    QIS = QuadrotorIsaacSim(CONFIG, task_name=None)
    for i in range(100):
        QIS.get_all_prim_names()
        time.sleep(3)
    QIS.stop_task()
