#
# This script create a customized Lidar sensor
# based on the range_sensor and 
# omni.kit.commands.execute of rotating lidar

# #
import time
import numpy as np
import omni
from typing import Optional, Tuple, List
import open3d as o3d

from omni.isaac.range_sensor import _range_sensor
from pxr import Gf, PhysicsSchemaTools, Sdf, Semantics, UsdGeom, UsdLux, UsdPhysics

class RotatingLidar():
    def __init__(
        self,
        config,
        number: Optional[int] = 0,
        rotation_rate: Optional[float] = None,
        translation: Optional[Tuple[float, float, float]] = None,
        orientation: Optional[np.ndarray] = None,
        fov: Optional[Tuple[float, float]] = None,
        resolution: Optional[Tuple[float, float]] = None,
        valid_range: Optional[Tuple[float, float]] = None,
        visualization: Optional[bool] = True,
    ) -> None:

        self.number = number
        self.update_config(config)
        self.set_lidar()
        self.accumulated_point_clouds = np.empty(shape=(0,3))
        self.temporary_point_clouds = np.empty(shape=(0,3))
        self.accumulate_step_length = 3  # steps waiting to be displayed 

        return
    # 
    # def set_configuration(self,number,rotation_rate,translation,orientation,fov,resolution,valid_range):
    #     # path
    #     prim_path = f"/World/envs/env_{number}"
    #     self.parent = prim_path + "/Crazyflie/body"
    #     self.lidarPath = "/lidar"
    #     self.lidar_path = self.parent+self.lidarPath

    #     # set default values
    #     # refer to RoboSense LiDar
    #     self.min_range, self.max_range = (0.4, 250.0) if valid_range is None else valid_range
    #     self.horizontal_fov, self.vertical_fov = (360.0, 70.0) if fov is None else fov
    #     self.horizontal_resolution, self.vertical_resolution = (0.2, 0.4) if resolution is None else resolution
    #     self.rotation_rate = 0.0 if rotation_rate is None else rotation_rate    # assume 10 Hz
    #     self.orientation = 0.0 if orientation is None else orientation
    #     self.translation = (0.0, 0.0, 0.05) if translation is None else translation

    #     # flags
    #     self.draw_lines = False
    #     self.display_points = True
    #     self.multi_line_mode = True
    #     self.enable_semantics = True
    #     return

    def update_config(self, config):
        '''
        config = config['lidar']
        '''
        prim_path = f"/World/envs/env_{self.number}"
        self.parent = prim_path + "/Crazyflie/body"
        self.lidarPath = "/lidar"
        self.lidar_path = self.parent+self.lidarPath

        self.min_range, self.max_range = config['range']
        self.horizontal_fov, self.vertical_fov = config['fov']
        self.horizontal_resolution, self.vertical_resolution = config['resolution']
        self.rotation_rate = config['rotation_rate']
        self.orientation = config['orientation']
        self.translation = config['translation']
        
        # flags
        self.draw_lines = config['draw_lines']
        self.display_points = config['display_points']
        self.multi_line_mode = config['multi_line_mode']
        self.enable_semantics = config['enable_semantics']
        self.visualization = config['visualization']
        return

    def set_lidar(self):
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface()     
        self.number_lasers = np.floor(self.vertical_fov/self.vertical_resolution).astype(int)+1
        self.physics_step = 0.0
        self.current_time = 0.0
        self.start_time = time.time()

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
        self.lidar_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(self.translation[0], 
                                                                       self.translation[1], 
                                                                       self.translation[2]))
        
        self.lidar_prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(self.orientation[0], 
                                                                       self.orientation[1], 
                                                                       self.orientation[2]))
        # from omni.isaac.core.prims import XFormPrim
        # from scipy.spatial.transform import Rotation as R
        # lidar_prim = XFormPrim(prim_path=self.lidar_path)
        # rotation_downward = R.from_euler('y', -90, degrees=True).as_quat()
        # translation, _ = lidar_prim.get_local_pose()  
        # lidar_prim.set_local_pose(translation, rotation_downward) 
        
        # set up the frame
        self._current_frame = dict()
        self._current_frame["time"] = 0
        self._current_frame["physics_step"] = 0
        self._current_frame["point_cloud"] = None
        self._current_frame["intensity"] = None
        self._current_frame["depth"] = None
        
        return
    
    def get_lidar_data(self, step_size: float):
        # self.physics_step += 1
        # self.time = time.time() - self.current_time
        
        # depth_points = np.empty((0,self.number_lasers,3))
        self.depth_points = self.lidarInterface.get_point_cloud_data(self.lidar_path)
        self.depth_points = np.squeeze(self.depth_points)

        if self.physics_step % self.accumulate_step_length:
            self.temporary_point_clouds = np.append(self.temporary_point_clouds, self.depth_points.reshape(-1,3), axis=0)
        else:
            self.accumulated_point_clouds = self.temporary_point_clouds
            self.temporary_point_clouds = np.empty(shape=(0,3))

        # type: numpy narray; shape: point_num, laser_num, coordinate
        

        if (self.physics_step % self.accumulate_step_length == 0) and self.visualization:
            if self.depth_points.shape[-1]==3 and self.depth_points.shape[0]!=0:
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(self.accumulated_point_clouds)
                # point_cloud.points = o3d.utility.Vector3dVector(depth_points.reshape(-1,3))
                # visualize the point cloud
                o3d.visualization.draw_geometries([point_cloud])
        
        self._data_acquisition_callback(step_size)

        return
    
    def _data_acquisition_callback(self, step_size: float) -> None:
        '''
        step_size is the time consumption in each simulation loop
        '''
        self.current_time += step_size
        self.physics_step += 1
        
        self._current_frame["intensity"] = self.lidarInterface.get_intensity_data(self.lidar_path)
        self._current_frame["depth"] = self.lidarInterface.get_depth_data(self.lidar_path)
        self._current_frame["physics_step"] = self.physics_step
        self._current_frame["time"] = self.current_time
        self._current_frame["point_cloud"] = self.depth_points

        if self.visualization:
            # print("current_frame intensity: ", self._current_frame["intensity"])
            # print("current_frame depth: ", self._current_frame["depth"])
            # print("current_frame physics_step: ", self._current_frame["physics_step"])
            # print("current_frame time: ", self._current_frame["time"])
            # print("current_frame LiDar data: ", self.depth_points[:5])
            # print("current_frame LiDar data shape: ", self.depth_points.shape)
            print("LiDar semantics: ", self.semantics[:5])
        return

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

    def resume(self) -> None:
        self._pause = False
        return

    def pause(self) -> None:
        self._pause = True
        return

    def is_paused(self) -> bool:
        return self._pause

    # def setup_lidars(self, scene):
    #     self.omni_lidar = scene.add(
    #         RotatingLidarPhysX(
    #             prim_path="/World/envs/env_0/Crazyflie/body/lidar", 
    #             name="lidar", 
    #             translation=np.array([0.0, 0.0, 0.05]),
    #             fov = (360.0, 45.0),
    #             resolution = (1.0, 5.0)
    #         )
    #     )
    #     self.omni_lidar.add_depth_data_to_frame()
    #     self.omni_lidar.add_point_cloud_data_to_frame()
    #     self.omni_lidar.enable_visualization()
    #     return

    # def get_pointcloud(self):
    #     # Retrieve the point cloud data from the annotator
        
    #     point_cloud_frame = self.omni_lidar.get_current_frame()
    #     point_cloud_data = point_cloud_frame['point_cloud']
    #     # print("LiDar frame: ", type(point_cloud_frame))
    #     # print("LiDar data: ", type(point_cloud_data))
        
    #     if point_cloud_data is not None:
    #         print("Frame keys: ", point_cloud_frame.keys())
    #         print("Key time: ", point_cloud_frame['time'])
    #         print("Key physics_step: ", point_cloud_frame['physics_step'])
    #         print("Key depth: ", point_cloud_frame['depth'][:5])
    #         print("Key point_cloud: ", point_cloud_frame['point_cloud'][:5])

    #         self.accumulated_point_clouds = np.append(self.accumulated_point_clouds, np.squeeze(point_cloud_data), axis=0)
    #         # 
    #         if point_cloud_frame['physics_step']%1==0:
    #             point_cloud = o3d.geometry.PointCloud()
    #             point_cloud.points = o3d.utility.Vector3dVector(self.accumulated_point_clouds)
    #             self.accumulated_point_clouds = np.empty(shape=(0,3))

    #             # visualize the point cloud
    #             o3d.visualization.draw_geometries([point_cloud])

    #     return point_cloud_data