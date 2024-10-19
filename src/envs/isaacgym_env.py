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

class PrimClass():
    def __init__(self, PrimMesh=None):
        self.set_mesh(PrimMesh)
        if PrimMesh:
            assert PrimMesh.IsValid() and PrimMesh.HasProperty('extent')
            self.calculate_attributes()
            self.calculate_position()

    def set_mesh(self, PrimMesh=None):
        self.mesh = PrimMesh
        self.isComplete = False

    def calculate_attributes(self):
        # Get the basic information
        self.name = self.mesh.GetName()
        self.path = self.mesh.GetPath().pathString
        # Get the categories

    def calculate_position(self):
        # Get its parent Xform of this mesh
        self.parent = self.mesh.GetParent()
        # Get the translate of its parent
        parent_path = self.parent.GetPath().pathString
        # print("prim name: ", cp_path) # Show full name
        import omni.isaac.core.utils.prims as prims_utils
        translation = prims_utils.get_prim_attribute_value(parent_path, attribute_name="xformOp:translate")
        
        # Get its outline, extent values 
        extent_attr = self.mesh.GetAttribute('extent')
        bbox = extent_attr.Get()
        print("The bounding box: ", bbox)
        # extent_attr may be None !
        if bbox:
            assert len(bbox) == 2
            self.bbox_min,self.bbox_max = bbox
            # Calculate the size
            self.size = self.bbox_max - self.bbox_min
            # Calculate the center
            local_center = list((self.bbox_max + self.bbox_min)/2)
            
            # calculate the position
            self.position = translation + local_center
            self.isComplete = True
        else:
            self.isComplete = False

    def show(self):
        print(f"The mesh path of Xform prim: {self.path}")
        print(f"The size of prim: {self.size}")
        print(f"The bounding box of prim: {self.bbox_min}, {self.bbox_max}")
        print(f"The position of prim: {self.position}")

class QuadrotorIsaacSim():
    """
    Initializes the QuadrotorIsaacSim environment.

    Args:
        visualizer: interface, optional.

    Attributes:
        _prev_arm_action (numpy.ndarray): Stores the previous arm action.
    """
    def __init__(self, config, usd_path=None):
        # Simple example showing how to start and stop the helper
        self.App = SimulationApp(launch_config=config)
        self.load_task(usd_path)
        time.sleep(2)

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
    
    def stop_task(self):
        self.App.close()  # Cleanup application
        sys.exit()

    def update(self):
        self.App.update()

    def is_running(self):
        return self.App._app.is_running() and not self.App.is_exiting()

    def get_all_prim_names(self):
        """
        Returns the names of all objects in the current task environment.

        Returns:
            list: A list of object names.
        """
        import omni.isaac.core.utils.prims as prims_utils
        # Get all prims
        predicate = lambda path: True
        child_prims = prims_utils.get_all_matching_child_prims("/", predicate)
        # print("child_prim: ", child_prims)
        # show the attributes of each child prim
        # print("prim attributes: ", dir(child_prims[7]))

        # Store child prims in dictionary
        primname_dict = {}
        for prim in child_prims:
            # Ignore the prims with '/Looks/' and single '/'
            #if '/Looks/' not in cp_path and cp_path.count('/') > 1:
            #    primname_dict[cp_path] = prim_position

            # store the prims with extent outlines
            if prim.IsValid() and prim.HasProperty('extent'):
                print("In prim:  ", prim)
                prim_element = PrimClass(prim)
                #prim_element.show()
                prim_name = prim_element.name
                if prim_element.isComplete:
                    primname_dict[prim_name] = prim_element
            else:
                continue
                print("Prim has NO extent property! Not stored!")
        print("All prim dictionary: ", primname_dict)

        return primname_dict
    
    def to_world(self, vertex, transform_matrix):
        # 将局部坐标转为世界坐标
        vertex_homogeneous = np.append(vertex, 1)  # 将顶点变为齐次坐标形式
        return np.dot(transform_matrix, vertex_homogeneous)[:3]
    
    def is_point_in_polygon(self, point_proj, poly_world_vertex):
        # number of vertices
        num_vertices = len(poly_world_vertex)

        angle_sum = 0.0

        for i in range(num_vertices):
            v1 = poly_world_vertex[i] - point_proj
            v2 = poly_world_vertex[(i + 1) % num_vertices] - point_proj

            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            dot_product = np.dot(v1, v2)
            angle = np.arccos(dot_product / (v1_norm * v2_norm))
            angle_sum += angle

        return np.isclose(angle_sum, 2 * np.pi)
    
    def point_to_plane_distance(self, point, poly_world_vertex):
        """
        calculate the distance from point to plane
        point: numpy array, include the coordinate (x, y, z)
        poly_world_vertex: contain numpy array, all vertices of the plane
        """
        p0, p1, p2 = poly_world_vertex[0], poly_world_vertex[1], poly_world_vertex[2]

        # get two vectors
        vec1 = p1 - p0
        vec2 = p2 - p0
        # Calculate the plane normal vector (cross product of vec1 and vec2)
        normal = np.cross(vec1, vec2)
        normal = normal / np.linalg.norm(normal)
        
        # calculate the D in plane Ax + By + Cz + D = 0
        D = -np.dot(normal, p0)
        
        # calculate the distance from point to plane
        distance = np.abs(np.dot(normal, point) + D) / np.linalg.norm(normal)

        # get the normal projection
        point_proj = point - (np.dot(normal, point) + D) * normal
        
        is_inside = self.is_point_in_polygon(point_proj, poly_world_vertex)

        return distance, is_inside

    def get_collision_prim(self, prim, point):
        """
        Judge whether point inside the prim
        return two flags
        1st flag: whether the prim has convex hull
        2nd flag: whether point inside the prim
        """
        from omni.physx import get_physx_cooking_interface, get_physx_interface
        from pxr import UsdGeom, Usd
        
        get_physx_interface().force_load_physics_from_usd()
        
        cooking_interface = get_physx_cooking_interface()

        prim_path = str(prim.GetPath())
        # print("prim_path: ", prim_path)
        transform_matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        position = transform_matrix.ExtractTranslation()
        print("position: ", position)
        num_convex_hulls = cooking_interface.get_nb_convex_mesh_data(prim_path)
        
        if num_convex_hulls == 0:
            return False, False

        distances_flags = []
        for hull_index in range(num_convex_hulls):
            convex_hull_data = cooking_interface.get_convex_mesh_data(prim_path, hull_index)
            # get vertices & polygons
            vertices = convex_hull_data["vertices"]
            polygons = convex_hull_data["polygons"]
            
            for poly_index in range(convex_hull_data["num_polygons"]):
                index_base = polygons[poly_index]["index_base"]
                # print("poly_index: ", poly_index)
                
                poly_world_vertex = []
                for vertex_index in range(polygons[poly_index]["num_vertices"]):
                    current_index = convex_hull_data["indices"][index_base + vertex_index]
                    poly_world_vertex.append(self.to_world(vertices[current_index], transform_matrix))
                # if "wall_side" in prim_path:
                #     print("prim_path: ", prim_path)
                #     print("poly_world_vertex: ", poly_world_vertex)
                    
                distance, is_inside_poly = self.point_to_plane_distance(point, poly_world_vertex)
                if is_inside_poly:
                    return True, True
                distances_flags.append((distance,is_inside_poly))
                
        # print("distances_flags: ", distances_flags)
        min_distance, inside_poly = min(distances_flags, key=lambda x: x[0])
        # print("min_distance: ", min_distance, "inside_poly?: ", inside_poly)

        return True, inside_poly
    
    def get_bounding_box(self, prim):
        """ return min and max coordinates of bounding box """
        from pxr import UsdGeom, Usd
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        box = bbox.ComputeWorldBound(prim).GetRange()
        return box.GetMin() / 100.0, box.GetMax() / 100.0

    def is_masked_prim(self, prim_path):
        masked_prims = ['Looks','Meshes','Lighting','Road','Buildings','Pavement',
                        'GroundPlane','TrafficLight','Bench','Tree','TableChair',
                        'Billboard','Lamp','RoadBarriers','Booth','Umbrella','Camera']
        
        for masked in masked_prims:
            if masked in prim_path:
                return True
        return False

    def save_all_prims_in_grid(self):
        from pxr import Gf
        # get current stage
        stage = omni.usd.get_context().get_stage()

        self.grid_resolution = 1.0 # less than 1.0 and can divide 1.0
        gr = self.grid_resolution
        
        self.prim_grid = {}
        
        # get all prims from the stage
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            # check whether prim_path is masked
            if self.is_masked_prim(prim_path):
                continue

            print("Prim: ", prim)
            min_bbox, max_bbox = self.get_bounding_box(prim)
            print("bounding_box: ", min_bbox, max_bbox)
            
            is_convex_hull, _ = self.get_collision_prim(prim, np.array(min_bbox))
            print("is_convex_hull: ", is_convex_hull)
            if is_convex_hull:
                x_range = np.arange(math.floor(min_bbox[0] / gr) * gr, math.ceil(max_bbox[0] / gr) * gr + gr, gr).tolist()
                y_range = np.arange(math.floor(min_bbox[1] / gr) * gr, math.ceil(max_bbox[1] / gr) * gr + gr, gr).tolist()
                z_range = np.arange(math.floor(min_bbox[2] / gr) * gr, math.ceil(max_bbox[2] / gr) * gr + gr, gr).tolist()

                for x in x_range:
                    for y in y_range:
                        for z in z_range:
                            point = np.array([x, y, z])
                            _, inside_poly = self.get_collision_prim(prim, point)
                            if inside_poly:
                                point_key = (x, y, z)
                                if point_key not in self.prim_grid:
                                    self.prim_grid[point_key] = prim
        # print("prim_grid: ", self.prim_grid)
        self.show_grid_by_prim(self.prim_grid)
        return self.prim_grid

    def get_prim_from_position(self, point):
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
