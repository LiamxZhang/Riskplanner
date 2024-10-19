#
#
# This script create a customized Lidar sensor
# based on the range_sensor and 
# rotating lidar physX

import sys
sys.path.append("..")
import numpy as np
import torch
import omni
from typing import Optional, List
from simple_pid import PID

from omni.isaac.core.tasks import BaseTask
from omni.isaac.cloner import GridCloner

from omni.isaac.core.utils.torch.rotations import *
import omni.isaac.core.utils.numpy.rotations as rot_utils

from pxr import Gf, UsdGeom, UsdLux
from robots.quadrotor import Quadrotor
from robots.quadrotor_view import QuadrotorView
from sensors.lidar import RotatingLidar
from sensors.camera import DepthCamera


class QuadrotorTask(BaseTask):
    def __init__(
        self,
        config
    ):
        # set robot name to "crazyflie"
        self._name = config['robot']['robot_name']
        super().__init__(name=self._name, offset=None)
        self.update_config(config)

        # velocity controller
        self.pid_controller = [PID(1, 0.1, 0.05),
                               PID(1, 0.1, 0.05),
                               PID(1, 0.1, 0.05),
                               PID(1, 0.1, 0.05),
                               PID(1, 0.1, 0.05),
                               PID(1, 0.1, 0.05)]
        return

    def update_config(self, config):
        self.config = config

        self._crazyflie_position = torch.tensor(config['robot']['init_position']) 
        self._crazyflie_orientation = torch.tensor(config['robot']['init_orientation'])
        self.num_observations = config['robot']['num_observations']

        self._num_envs = config['env']['num_envs']
        self._env_spacing = config['env']['env_spacing']

        self.obs_buf = torch.zeros((self._num_envs, self.num_observations))
        
        self.dof_vel = torch.zeros((self._num_envs, 4))
        self.all_indices = torch.arange(self._num_envs)

        self.accumulated_point_clouds = np.empty(shape=(0,3))

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        # add quadrotor to the scene
        self.default_zero_env_path = "/World/envs/env_0"
        
        self.copter = Quadrotor(
                    prim_path=self.default_zero_env_path + "/Crazyflie", name=self._name, 
                    translation=self._crazyflie_position, 
                    orientation=rot_utils.euler_angles_to_quats(self._crazyflie_orientation, degrees=True)
                )
        
        # add robot view
        self._copters_view = QuadrotorView(prim_paths_expr="/World/envs/.*/Crazyflie", name="crazyflie_view")
        scene.add(self._copters_view)
        for i in range(4):
            scene.add(self._copters_view.physics_rotors[i])

        # add camera
        self.camera = DepthCamera(config=self.config['camera'],visualization=True)
        # add Lidar
        self.lidar = RotatingLidar(config=self.config['lidar_downward'],visualization=True)
        return

    def initialize_views(self, scene):
        default_base_env_path = "/World/envs"
        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(default_base_env_path)

        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, self.default_zero_env_path)
        
        prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
        self._env_pos = self._cloner.clone(
            source_prim_path="/World/envs/env_0", prim_paths=prim_paths, replicate_physics=True, copy_from_source=False
        )
        
        pos, _ = self._cloner.get_clone_transforms(self._num_envs)


    def get_observations(self):
        self.root_pos, self.root_rot = self._copters_view.get_world_poses(clone=False)
        # self.root_velocities = self._copters_view.get_linear_velocities()
        # self.dof_vel = self._copters_view.get_joint_velocities()
        # print("root_velocities: ", self.root_velocities)

        # To Torch Tensor
        self.root_positions = torch.from_numpy(self.root_pos)
        root_quats = torch.tensor(self.root_rot) # shape: 2*4
        self.root_orient = self.quat_to_euler(root_quats)
        #
        # print("position feedback: ", self.root_positions)
        # print("orientation feedback: ", self.root_orient)

        rot_x = quat_axis(root_quats, 0) # coordinates transformed to new x
        rot_y = quat_axis(root_quats, 1) # coordinates transformed to new y
        rot_z = quat_axis(root_quats, 2) # coordinates transformed to new z

        # root_linvels = self.root_velocities[:, :3]
        # root_angvels = self.root_velocities[:, 3:]

        self.obs_buf[..., 0:3] = self.root_positions 

        self.obs_buf[..., 3:6] = rot_x
        self.obs_buf[..., 6:9] = rot_y
        self.obs_buf[..., 9:12] = rot_z

        # self.obs_buf[..., 12:15] = root_linvels
        # self.obs_buf[..., 15:18] = root_angvels

        observations = {self._copters_view.name: {"obs_buf": self.obs_buf}}
        # print("observations: ", observations)
        return observations

    def velocity_action(self, actions) -> None:
        # actions: 1*6
        # kinematics-based control 

        # if not self.world.is_playing():
        #     return
        self.actions = torch.tensor(actions).unsqueeze(0)
        # print("actions: ", actions)

        # clamp to [-1.0, 1.0] 
        vel_cmds = torch.clamp(self.actions, min=-1.0, max=1.0)
        # print("vel_cmds: ", vel_cmds)

        self.velocity_max = torch.tensor([2, 2, 2])
        linear_velocities = self.velocity_max * vel_cmds[...,:3]
        
        # spin spinning rotors
        self.prop_max_rot = 4333
        prop_rot = torch.abs(vel_cmds * self.prop_max_rot)
        # print("prop_rot: ", prop_rot)
        self.dof_vel[:, 0] = prop_rot[:, 2]
        self.dof_vel[:, 1] = -1.0 * prop_rot[:, 2]
        self.dof_vel[:, 2] = prop_rot[:, 2]
        self.dof_vel[:, 3] = -1.0 * prop_rot[:, 2]
        
        self._copters_view.set_joint_velocities(self.dof_vel)
        # print("rotor spinning velocities: ", self.dof_vel)
        
        self.angular_vel_max =  10 * np.pi/2 * torch.tensor([1, 1, 1])
        angular_velocities = self.angular_vel_max * vel_cmds[...,3:]

        self.velocities = torch.cat((linear_velocities, angular_velocities), dim=1)
        # print("velocity commands: ", self.velocities)

        # apply actions
        self._copters_view.set_velocities(velocities=self.velocities,indices=torch.tensor([0]))

    def quat_to_euler(self, root_quats):
        """
        Converts rotation matrix to Euler angles (ZYX order).
        R: Rotation matrix, shape (N, 3, 3)
        Returns: Euler angles (roll, pitch, yaw), shape (N, 3)
        """
        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)

        R = torch.stack((rot_x, rot_y, rot_z), dim=2)

        sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
        singular = sy < 1e-6

        x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        y = torch.atan2(-R[:, 2, 0], sy)
        z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

        x[singular] = torch.atan2(-R[singular, 1, 2], R[singular, 1, 1])
        y[singular] = torch.atan2(-R[singular, 2, 0], sy[singular])
        z[singular] = 0

        return torch.stack((x, y, z), dim=1)
    
    def controller(self, target_position, target_orient):
        # inputs are all 1*3 array
        output = []
        current_position = np.squeeze(self.root_pos)
        for i in range(3):
            self.pid_controller[i].setpoint = target_position[i]
            cmd = self.pid_controller[i](current_position[i])
            output.append(cmd)

        current_orient = np.squeeze(np.array(self.root_orient))
        for i in range(3):
            self.pid_controller[i+3].setpoint = target_orient[i]
            cmd = self.pid_controller[i+3](current_orient[i])
            output.append(cmd)

        # print("PID controller output: ", output)
        return output
    
    def force_action(self, propulsion) -> None:
        return
    

    def lidar_local2world(self):
        '''
        Transform the lidar data to world coordinate
        '''
        # check each point cloud
        self.lidar_data = self.lidar.depth_points.copy() *0.1428  # / 7
        # print("LiDar data: ", self.lidar_data[:5])

        # get the position and orientation of quadrotor in global frame (torch.tensor)
        drone_position = self.root_positions 
        drone_euler = self.root_orient 

        # get the translation and orientation of lidar in quadrotor frame (torch.tensor)
        lidar_translation = torch.tensor(self.lidar.translation, dtype=torch.float32)  
        lidar_euler = self.lidar.orientation   

        # Rotation matrices
        from scipy.spatial.transform import Rotation as R
        drone_rotation_matrix = torch.tensor(R.from_euler('xyz', drone_euler.tolist(), degrees=True).as_matrix(), dtype=torch.float32)
        lidar_rotation_matrix = torch.tensor(R.from_euler('xyz', lidar_euler, degrees=True).as_matrix(), dtype=torch.float32)

        if False:
            print("Drone translation matrix: ", drone_position)
            print("Drone rotation angle: ", drone_rotation_matrix)
            print("Lidar translation matrix: ", lidar_translation)
            print("Lidar rotation angle: ", lidar_rotation_matrix)

        # Take the lower half of point cloud
        lidar_data_full = torch.tensor(self.lidar_data, dtype=torch.float32)
        row, col, _ = lidar_data_full.shape
        # print("Before LiDAR data shape: %d and %d" %(row, col))

        N = 3
        part_col = col * (N-1) // N
        lidar_data_part = lidar_data_full[:, part_col:, :]
        # print("Half LiDAR data: ", lidar_data_part[:5])
        row, col, _ = lidar_data_part.shape
        # print("After LiDAR data shape: %d and %d"%(row, col))

        # Transform local frame to global frame
        self.lidar_data_in_world = torch.empty_like(lidar_data_part)
        for i in range(row):
            for j in range(col):
                # Get the lidar data point from the lidar_data_part
                lidar_points = torch.tensor(lidar_data_part[i][j], dtype=torch.float32) 

                # Transform to drone coordinate
                lidarpoint_in_drone_frame = torch.matmul(lidar_rotation_matrix, lidar_points.T).T + lidar_translation
                # Transform to global coordinate
                lidarpoint_in_world_frame = torch.matmul(drone_rotation_matrix, lidarpoint_in_drone_frame.T) + drone_position

                self.lidar_data_in_world[i, j, :] = lidarpoint_in_world_frame
        # display the lidar in world
        # print("LiDAR data in world frame: ", lidar_data_in_world[:5])

        return
    
    def get_lidar_data(self):
        target_position = Gf.Vec3d(self.lidar_data_in_world[0, 0, 0].item(), 
                                   self.lidar_data_in_world[0, 0, 1].item(), 
                                   self.lidar_data_in_world[0, 0, 2].item())

        target_grid_key = [target_position[0], target_position[1], target_position[2]]
        # print("lidar point: ", target_grid_key)
        return target_grid_key

    def object_detection(self, prim_grid, grid_size):
        '''
        According to the position of each point cloud, determine its semantic information and classification
        prim_grid: 
        grid_size: 
        '''
        # lidar_data_in_world

        # detection position from self.lidar_data_in_world
        target_position = Gf.Vec3d(self.lidar_data_in_world[0, 0, 0].item(), 
                                   self.lidar_data_in_world[0, 0, 1].item(), 
                                   self.lidar_data_in_world[0, 0, 2].item())

        # 设置一个阈值
        threshold = 0.1 * grid_size

        # 查找目标位置所属的网格
        target_grid_key = (int(target_position[0] // grid_size),
                        int(target_position[1] // grid_size),
                        int(target_position[2] // grid_size))

        # 查找与目标网格及其邻近网格中的 Prims
        closest_prim = None
        closest_distance = float('inf')

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    neighbor_grid_key = (target_grid_key[0] + dx,
                                        target_grid_key[1] + dy,
                                        target_grid_key[2] + dz)
                    if neighbor_grid_key in prim_grid:
                        for prim, prim_position in prim_grid[neighbor_grid_key]:
                            # 计算 Prim 与目标位置之间的距离
                            distance = np.linalg.norm(np.array(prim_position) - np.array(target_position))

                            # 如果找到更接近的 Prim，则更新
                            if distance < closest_distance and distance < threshold:
                                closest_distance = distance
                                closest_prim = prim

        # 输出最接近目标位置的 Prim 信息
        if closest_prim:
            print(f"Closest prim found: {closest_prim.GetPath()}")
        else:
            print("No prim found near the given position.")

        return