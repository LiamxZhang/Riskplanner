# This script create a customized quadrator task
# 
# 

import sys
sys.path.append("..")
import math
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
from controller.min_snap_traj import SnapTrajectory
from controller.nonlinear_controller import NonlinearController

class QuadrotorTask(BaseTask):
    def __init__(self,config):
        self._name = config['robot']['robot_name']  # set robot name to "crazyflie"
        super().__init__(name=self._name, offset=None)
        self.init_config(config)

        # velocity controller
        self.pid_controller = [PID(1, 0.1, 0.0),
                               PID(1, 0.1, 0.0),
                               PID(1, 0.1, 0.0),
                               PID(1, 0.0, 0.0),
                               PID(1, 0.0, 0.0),
                               PID(1, 0.0, 0.0)]
        
        

    def init_config(self, config):
        self.config = config

        self._crazyflie_position = torch.tensor(config['robot']['init_position']) 
        self._crazyflie_orientation = torch.tensor(config['robot']['init_orientation'])
        self.num_observations = config['robot']['num_observations']

        self._num_envs = config['env']['num_envs']
        self._env_spacing = config['env']['env_spacing']

        self.obs_buf = torch.zeros((self._num_envs, self.num_observations))
        
        self.dof_vel = torch.zeros((self._num_envs, 4))
        self.all_indices = torch.arange(self._num_envs)

        self.accumulated_point_clouds = torch.empty((0, 3), dtype=torch.float32)


    def set_up_scene(self, scene):
        """ setup the isaac sim scene and load the quadrotor"""
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
        # add traj planner
        self.trajplanner = SnapTrajectory(7)
        # add controller
        self.nlcontroller = NonlinearController(default_env_path=self.default_zero_env_path)

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

    def start(self):
        self.get_observations()
        self.nlcontroller.start(-1*self.config['env']['gravity'][2])
    
    def get_observations(self):
        # Position information
        root_pos, root_rot = self._copters_view.get_world_poses(clone=False)
        self.root_positions = torch.from_numpy(root_pos)
        root_quats = torch.tensor(root_rot) # root_rot shape: 2*4
        self.root_orient = self.quat_to_euler(root_quats) # Radian degree
        # print("Position and orientation feedback: ", self.root_positions, self.root_orient)

        # Velocity information
        try:
            root_velocities = self._copters_view.get_velocities(clone=False)
            self.root_linvels = torch.from_numpy(root_velocities[:, :3])
            self.root_angvels = torch.from_numpy(root_velocities[:, 3:])
        except:
            self.root_linvels = torch.zeros((1, 3))  
            self.root_angvels = torch.zeros((1, 3))
        # self.dof_vel = self._copters_view.get_joint_velocities()
        # print("root_velocities: ", self.root_linvels, self.root_angvels)

        # Save to [observations]
        self.obs_buf[..., 0:3] = self.root_positions 
        self.obs_buf[..., 3:6] = quat_axis(root_quats, 0) # coordinates transformed to new x
        self.obs_buf[..., 6:9] = quat_axis(root_quats, 1) # coordinates transformed to new y
        self.obs_buf[..., 9:12] = quat_axis(root_quats, 2) # coordinates transformed to new z
        self.obs_buf[..., 12:15] = self.root_linvels
        self.obs_buf[..., 15:18] = self.root_angvels
        observations = {self._copters_view.name: {"obs_buf": self.obs_buf}}
        # print("observations: ", observations)

        return self.root_positions, self.root_orient, self.root_linvels, self.root_angvels

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
        
        self.angular_vel_max =  2 * math.pi * torch.tensor([1, 1, 1])
        angular_velocities = self.angular_vel_max * vel_cmds[...,3:]

        self.velocities = torch.cat((linear_velocities, angular_velocities), dim=1)
        # print("velocity commands: ", self.velocities)

        # apply actions
        self._copters_view.set_velocities(velocities=self.velocities,indices=torch.tensor([0]))

    def thrust_action(self, thrust) -> None:
        self.actions = thrust.unsqueeze(0)
        
        ## spin rotors
        # clamp to [-1.0, 1.0] 
        vel_cmds = torch.clamp(self.actions, min=-1.0, max=1.0)

        self.prop_max_rot = 4333
        prop_rot = torch.abs(vel_cmds * self.prop_max_rot)
        # print("prop_rot: ", prop_rot)
        self.dof_vel[:, 0] = prop_rot[:, 0]
        self.dof_vel[:, 1] = -1.0 * prop_rot[:, 1]
        self.dof_vel[:, 2] = prop_rot[:, 2]
        self.dof_vel[:, 3] = -1.0 * prop_rot[:, 3]
        
        self._copters_view.set_joint_velocities(self.dof_vel)
        # print("rotor spinning velocities: ", self.dof_vel)

        # apply actions
        for i in range(4):
            self._copters.physics_rotors[i].apply_forces(self.thrusts[:, i], indices=self.all_indices)
        return

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
        target_orient = [math.radians(angle) for angle in target_orient]
        output = []
        current_position = torch.squeeze(self.root_positions)

        for i in range(3):
            self.pid_controller[i].setpoint = target_position[i]
            cmd = self.pid_controller[i](current_position[i])
            output.append(cmd)

        current_orient = torch.squeeze(torch.tensor(self.root_orient))

        for i in range(3):
            self.pid_controller[i+3].setpoint = target_orient[i]
            cmd = self.pid_controller[i+3](current_orient[i])
            output.append(cmd)
        # print("PID feedback: ", current_position, current_orient)
        # print("PID controller target: ", target_position, target_orient)
        # print("PID controller output: ", output)
        return output
    
    def control_update(self, dt, targets):
        '''
        Method that implements the nonlinear control law. 
        This method will be called by the simulation on every physics step.

        dt: time step during a control loop
        targets: all target points to be tracked
        '''
        # # Current states: self.root_positions, self.root_orient, self.root_linvels, self.root_angvels
        state = {"position":self.root_positions, 
                 "attitude":self.root_orient,
                 "angular_velocity":self.root_angvels,
                 "linear_velocity":self.root_linvels}
        self.nlcontroller.update_state(state)
        print("state: ", state)
        
        # # Organize target waypoints
        waypoints = []
        x_c, y_c, z_c = self.root_positions[0].tolist()
        psi_c = self.root_orient[0, 2].item()
        waypoints.extend([[x_c], [y_c], [z_c], [psi_c], [0.0]])

        for i, point in enumerate(targets, start=1):
            x, y, z, psi = point
            t = i * dt # dt
            waypoints.extend([[x], [y], [z], [math.radians(psi)], [t]])
        # print("All waypoints: ", waypoints)
        
        # # Compute target trajectory
        try:
            self.trajplanner.traj(waypoints)
            traj_target = self.trajplanner.output_next_traj_point()
        except:
            traj_target= {
                            "position": [x_c, y_c, z_c],       
                            "velocity": [0,0,0],       
                            "acceleration": [0,0,0], 
                            "jerk": [0,0,0],        
                            "yaw_angle": psi_c, 
                            "yaw_rate": 0.0       
                        }
        print("Next traj_target: ", traj_target)
        
        # Update controller
        self.nlcontroller.update_trajectory(traj_target)

        # Control update
        u_1, tau = self.nlcontroller.update(dt)
        rotor_thrusts = self.nlcontroller.force_and_torques_to_thrust(u_1, tau)
        
        return rotor_thrusts

    def lidar_local2world(self):
        '''
        Transform the lidar data to world coordinate
        '''
        # check each point cloud
        self.lidar_data = self.lidar.depth_points.copy() *0.1428  # divided by 7
        print("LiDar raw data: ", self.lidar_data[:5])

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
        print("LiDAR data in world frame: ", self.lidar_data_in_world[:5])

        return
    
    def get_lidar_data(self):
        target_position = Gf.Vec3d(self.lidar_data_in_world[0, 0, 0].item(), 
                                   self.lidar_data_in_world[0, 0, 1].item(), 
                                   self.lidar_data_in_world[0, 0, 2].item())

        target_grid_key = [target_position[0], target_position[1], target_position[2]]
        # print("lidar point: ", target_grid_key)
        return target_grid_key

    def object_detection(self, prim_grid, grid_resolution):
        '''
        According to the position of each point cloud, determine its semantic information and classification
        prim_grid: grid world map created by isaacgym_env library
        grid_resolution: grid world resolution in the grid map
        '''
        gr = grid_resolution
        grid_coord = (
            round(point[0] / gr) * gr,
            round(point[1] / gr) * gr,
            round(point[2] / gr) * gr
        )
        # print("grid_coord: ", grid_coord)
        if grid_coord in self.prim_grid:
            return self.prim_grid[grid_coord]  # 返回对应的 prim
        # lidar_data_in_world

        # detection position from self.lidar_data_in_world
        target_position = Gf.Vec3d(self.lidar_data_in_world[0, 0, 0].item(), 
                                   self.lidar_data_in_world[0, 0, 1].item(), 
                                   self.lidar_data_in_world[0, 0, 2].item())

        # 
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
                            distance = torch.norm(torch.tensor(prim_position) - torch.tensor(target_position))


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