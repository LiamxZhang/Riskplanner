#
# This script create a customized Lidar sensor
# based on the range_sensor and 
# omni.kit.commands.execute of rotating lidar

# Common APIs
import numpy as np
import torch
from typing import Optional, Tuple, List
from scipy.spatial.transform import Rotation as R
import sys
try:
    import open3d as o3d
except ImportError as e:
    import subprocess
    install_command = [sys.executable, "-m", "pip", "install", "open3d"]
    result = subprocess.run(install_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    import open3d as o3d # May need to restart the program

# Isaac sim APIs
import omni
from omni.isaac.range_sensor import _range_sensor
from omni.usd import get_stage_next_free_path

from pxr import Gf, PhysicsSchemaTools, Sdf, Semantics, UsdGeom, UsdLux, UsdPhysics

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from utils.state import State
from sensors.graphical_sensor import GraphicalSensor
from envs.isaacgym_env import QuadrotorIsaacSim
from configs.configs import LIDAR_PARAMS


class RotatingLidar(GraphicalSensor):
    def __init__(self):
        # Setup the name of the camera primitive path
        self._lidar_name = "lidar"
        self._stage_prim_path = ""

        # Initialize the Super class "object" attributes
        super().__init__(sensor_type=LIDAR_PARAMS['type'], update_rate=LIDAR_PARAMS['frequency']) 

        self.accumulated_point_clouds = torch.empty((0, 3), dtype=torch.float32)
        self.temporary_point_clouds = torch.empty((0, 3), dtype=torch.float32)
        self.accumulate_step_length = 3  # num of steps to be accumulated 

        # Create the lidar interface
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface() # Used to interact with the LIDAR


    def update_config(self):
        '''
        config = config['lidar']
        '''
        # Prepare the primary/temporary lidar path
        self.parent = self._vehicle.stage_prefix + "/body"
        self.lidarPath = "/" + self._lidar_name
        self.lidar_primary_path = self.parent + self.lidarPath

        # Get the complete stage prefix for the lidar
        self._stage_prim_path = get_stage_next_free_path(QuadrotorIsaacSim().world.stage, self.lidar_primary_path, False)

        # Get the camera name that was actually created (and update the camera name)
        self._lidar_name = self._stage_prim_path.rpartition("/")[-1]

        # Parameters
        self.min_range, self.max_range = LIDAR_PARAMS['range']
        self.horizontal_fov, self.vertical_fov = LIDAR_PARAMS['fov']
        self.horizontal_resolution, self.vertical_resolution = LIDAR_PARAMS['resolution']
        self.rotation_rate = LIDAR_PARAMS['rotation_rate']
        self.orientation = torch.tensor(LIDAR_PARAMS['orientation'], dtype=torch.float32)
        self.translation = torch.tensor(LIDAR_PARAMS['translation'], dtype=torch.float32)
        
        # Flags
        self.draw_lines = LIDAR_PARAMS['draw_lines']
        self.display_points = LIDAR_PARAMS['display_points']
        self.multi_line_mode = LIDAR_PARAMS['multi_line_mode']
        self.enable_semantics = LIDAR_PARAMS['enable_semantics']
        self.visualization = LIDAR_PARAMS['visualization']

    def initialize(self, vehicle):
        """
        Initialize the Lidar sensor
        """
        super().initialize(vehicle)
        self.update_config()

        # Initialize deep point data frame
        self.reset()

        # Get lidar instance
        result, lidar = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=self.lidarPath,
            parent=self.parent,
            min_range=self.min_range,
            max_range=self.max_range,
            draw_points=self.display_points,
            draw_lines=self.draw_lines,
            horizontal_fov=self.horizontal_fov,
            vertical_fov=self.vertical_fov,
            horizontal_resolution=self.horizontal_resolution,
            vertical_resolution=self.vertical_resolution,
            rotation_rate=self.rotation_rate,
            high_lod=self.multi_line_mode,
            yaw_offset=0.0,
            enable_semantics=self.enable_semantics
        )

        self.lidar_prim = lidar.GetPrim()
        self.lidar_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(self.translation[0].item(), 
                                                                       self.translation[1].item(), 
                                                                       self.translation[2].item()))
        
        self.lidar_prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(self.orientation[0].item(), 
                                                                       self.orientation[1].item(), 
                                                                       self.orientation[2].item()))
    
    def get_lidar_data(self, time_step: float):
        # depth_points has a shape of (0,self.number_lasers,3) as a placeholder for the lidar data
        self.depth_points = self.lidarInterface.get_point_cloud_data(self._stage_prim_path) 
        self.depth_points = torch.tensor(self.depth_points, dtype=torch.float32)

        # For visualization
        if self.physics_step % self.accumulate_step_length:
            # self.temporary_point_clouds = np.append(self.temporary_point_clouds, self.depth_points.reshape(-1,3), axis=0)
            self.temporary_point_clouds = torch.cat(
                (self.temporary_point_clouds, self.depth_points.view(-1, 3)), dim=0)
        else: # exact division
            self.accumulated_point_clouds = self.temporary_point_clouds.clone()
            self.temporary_point_clouds = torch.empty((0, 3), dtype=torch.float32)
        # type: torch.tensor; shape: point_num, laser_num, coordinate
        if (self.physics_step % self.accumulate_step_length == 0) and self.visualization:
            if self.accumulated_point_clouds.size(0) != 0:
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(self.accumulated_point_clouds.numpy())
                # visualize the point cloud
                o3d.visualization.draw_geometries([point_cloud])
        
        self._data_acquisition_callback(time_step)
    
    def _data_acquisition_callback(self, time_step: float):
        '''
        time_step is the time consumption in each simulation loop
        '''
        self.total_time += time_step
        self.physics_step += 1
        
        # self._current_frame["intensity"] = self.lidarInterface.get_intensity_data(self.lidar_path)
        # self._current_frame["depth"] = self.lidarInterface.get_depth_data(self.lidar_path)
        # self._current_frame["physics_step"] = self.physics_step
        # self._current_frame["time"] = self.total_time
        # self._current_frame["point_cloud"] = self.depth_points
        self._current_frame["intensity"] = torch.tensor(
            self.lidarInterface.get_intensity_data(self._stage_prim_path), dtype=torch.float32)
        self._current_frame["depth"] = torch.tensor(
            self.lidarInterface.get_depth_data(self._stage_prim_path), dtype=torch.float32)
        self._current_frame["physics_step"] = self.physics_step
        self._current_frame["time"] = self.total_time
        self._current_frame["point_cloud"] = self.depth_points

    def lidar_local2world(self, drone: State):
        '''
        Transform the lidar data to world coordinate
        '''
        # Rotation matrices
        drone_rotation_matrix = torch.tensor(
            R.from_euler('ZYX', drone.orient.tolist(), degrees=False).as_matrix(), 
            dtype=torch.float32
            )
        lidar_rotation_matrix = torch.tensor(
            R.from_euler('ZYX', self.orientation.tolist(), degrees=False).as_matrix(), 
            dtype=torch.float32
            )
        # print(f"Rotation matrices: {drone_rotation_matrix} and {lidar_rotation_matrix}")
        
        # # To show full raw data in world frame
        # points = self.depth_points.view(-1,3)
        # lidar_points_in_drone_frame = torch.matmul(
        #     lidar_rotation_matrix, points.T
        # ).T + self.translation  # Local to drone frame

        # lidar_data_in_world = torch.matmul(
        #     drone_rotation_matrix, lidar_points_in_drone_frame.T
        # ).T + drone.position  # Drone to global frame
        # print("LiDar raw data shape:",self.depth_points.shape)
        # print("LiDar raw data: ", lidar_data_in_world)

        # Slice the tensor
        N = 1
        _, col, _ = self.depth_points.shape  # Get the width (horizontal resolution)
        part_col = col * (N - 1) // N  # Starting column index for bottom 1/N
        # Shape: (H, W/N, 3) -> Shape: (H*W/N, 3)
        lidar_data_part = self.depth_points[:, part_col:, :].contiguous().view(-1, 3)   

        # Batch processing: transform all points at once
        lidar_points_in_drone_frame = torch.matmul(
            lidar_rotation_matrix, lidar_data_part.T
        ).T + self.translation  # Local to drone frame
        self.lidar_data_in_world = torch.matmul(
            drone_rotation_matrix, lidar_points_in_drone_frame.T
        ).T + drone.position  # Drone to global frame
        self._current_frame["point_cloud"] = self.lidar_data_in_world
        # print("Sliced LiDar data shape: ", self.lidar_data_in_world.shape)
        # print("LiDAR data in world frame: ", self.lidar_data_in_world)
        

    @GraphicalSensor.update_at_rate
    def update(self, state: State, dt: float):
        """Method that gets the data from the lidar and returns it as a dictionary.

        Args:
            state (State): The current state of the vehicle.
            dt (float): The time elapsed between the previous and current function calls (s).

        Returns:
            (dict) A dictionary containing the current state of the sensor (the data produced by the sensor)
        """

        self.get_lidar_data(dt)
        # Just return the prim path and the name of the lidar
        # self._state = {"lidar_name": self._lidar_name, "stage_prim_path": self._stage_prim_path}
        self.lidar_local2world(state)

        return self._current_frame


    def check_config(self):
        if self.lidar_prim:
            print("Lidar Configuration:")
            print("  - Horizontal FOV: ", self.lidar_prim.GetAttribute("horizontalFov").Get())
            print("  - Vertical FOV: ", self.lidar_prim.GetAttribute("verticalFov").Get())
            print("  - Horizontal Resolution: ", self.lidar_prim.GetAttribute("horizontalResolution").Get())
            print("  - Vertical Resolution: ", self.lidar_prim.GetAttribute("verticalResolution").Get())
            print("  - Min Range: ", self.lidar_prim.GetAttribute("minRange").Get())
            print("  - Max Range: ", self.lidar_prim.GetAttribute("maxRange").Get())
            print("  - Rotation Rate: ", self.lidar_prim.GetAttribute("rotationRate").Get())
            print("  - Yaw Offset: ", self.lidar_prim.GetAttribute("yawOffset").Get())
        else:
            print("Error: no lidar_prim! ")

    def add_intensity_data_to_frame(self) -> None:
        self._current_frame["intensity"] = None
        return

    def remove_intensity_data_from_frame(self) -> None:
        del self._current_frame["intensity"]
        return
    
    def add_depth_data_to_frame(self) -> None:
        self._current_frame["depth"] = None
        return

    def remove_depth_data_from_frame(self) -> None:
        del self._current_frame["depth"]
        return

    @property
    def data(self):
        """
        (dict) The 'state' of the sensor, i.e. the data produced by the sensor at any given point in time
        """
        return self._current_frame
    
    @property
    def info(self):
        """
        (dict) The 'information' of the sensor, i.e. the name, path and configuration of the sensor
        """
        self._info = {"lidar_name": self._lidar_name, "stage_prim_path": self._stage_prim_path, "configuration": LIDAR_PARAMS}

        return self._info

    def reset(self):
        """Method that when implemented, should handle the reset of the vehicle simulation to its original state
        """
        self.total_time = 0.0
        self.physics_step = 0.0

        # set up the frame
        self._current_frame = dict()
        self._current_frame["time"] = torch.tensor(0.0, dtype=torch.float32)  
        self._current_frame["physics_step"] = torch.tensor(0, dtype=torch.int32)  
        self._current_frame["point_cloud"] = torch.empty((0, 3), dtype=torch.float32)  
        self._current_frame["intensity"] = torch.empty(0, dtype=torch.float32)  
        self._current_frame["depth"] = torch.empty(0, dtype=torch.float32) 
