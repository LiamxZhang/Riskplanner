#!/usr/bin/env python
import numpy as np
import torch
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
# Isaac Sim APIs
from envs.isaacgym_env import QuadrotorIsaacSim
from configs.configs import MAP_ASSET, ROBOT_PARAMS, CONTROL_PARAMS
from utils.task_util import spherical_to_cartesian

from controller.nonlinear_controller import NonlinearController
from map.sense_gridmap import SenseGridMap
from sensors.lidar import RotatingLidar
from robots.quadrotor import Quadrotor

# gym APIs
import gymnasium as gym
from gymnasium import spaces


class PerceptDroneNav(gym.Env):
    """Base class for "drone navigation" Gym environments."""
    # The isaac sim simulation environment has already been created
    # metadata = {"render_modes": ["human"], "render_fps": 30}

    ################################################################################

    def __init__(self, output_folder='results'):
        
        """Initialization of a generic aviary environment.
        """
        super(PerceptDroneNav, self).__init__()

        # Create the vehicle environment
        controller = NonlinearController(
                stage_prefix=ROBOT_PARAMS['stage_prefix'],
                Ki=[0.5, 0.5, 0.5],
                Kr=[2.0, 2.0, 2.0]
            )
        self.sense_gridmap = SenseGridMap()
        lidar = RotatingLidar()

        self.quadrotor = Quadrotor(**ROBOT_PARAMS, sensors=[], 
                                   graphical_sensors=[lidar], 
                                   backends=[controller,self.sense_gridmap])

        QuadrotorIsaacSim().reset() 

        # Define action and observation space
        # The action will be normalized as 2 angles in polar coordinates
        # and compute the x,y,z,psi coordinates
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # The observation will be the local grid map and relative state of the agent
        self.grid_map_dim = (1, 32, 32, 32)  # Dimension: [batch_size, length, width, height]
        self.state_dim = 6  # Dimension of all state: [x, y, z, vx, vy, vz]

        self.observation_space = spaces.Dict({
            "gridmap": spaces.Box(low=0, high=MAP_ASSET["max_fill_value"], shape=self.grid_map_dim, dtype=np.float32),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        })

        # Configurations
        self.target_position = (torch.tensor(CONTROL_PARAMS["target_position"], dtype=torch.float32)
                                - torch.tensor(ROBOT_PARAMS["init_position"], dtype=torch.float32))
        # param里定义的targetposition是相对于世界坐标系原点的
        # 但是self.quadrotor.state好像是相对于init_position原点的
        # 这个类似于airsim
        self.max_steps = 1e4
        self.current_step = 0

        self.distance = None
        self.last_distance = None
        self.is_flipover = False


    ################################################################################

    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        self.current_step = 0
        # Initialize random number generator with seed
        super().reset(seed=seed, options=options)
        # Reset the quadrotor state
        self.quadrotor.reset()
        # Return observation and info
        observation = {
            "gridmap": self._get_local_gridmap(), 
            "state": self._get_drone_state() 
        }

        return observation, {}  # empty info dict
    
    ################################################################################

    def step(self, action):
        """Advances the environment by one simulation step.
        """
        # Count 1 step
        self.current_step += 1
        # Update state
        self.state = self.quadrotor.state

        # Adjust action to a trajectory point 
        radius = 0.2
        x, y, z = spherical_to_cartesian(action, radius) # Target waypoint

        relative_position = self.target_position - self.state.position
        x = relative_position[0].item()
        y = relative_position[1].item()
        z = relative_position[2].item()

        x0 = self.state.position[0].item()  # Current x position
        y0 = self.state.position[1].item()  # Current y position
        z0 = self.state.position[2].item()  # Current z position

        x += x0  # Add current x position
        y += y0  # Add current y position
        z += z0  # Add current z position

        # Compute yaw angle using atan2
        psi = np.arctan2(y - y0, x - x0)
        psi += self.state.angular_velocity[2].item()  # Add current psi angle

        # Constrain psi to [-π, π]
        psi = (psi + np.pi) % (2 * np.pi) - np.pi

        # Apply new trajectory point to quadrotor
        for i in range(50):
            self.quadrotor.update_trajectory([[x, y, z, psi]])
            QuadrotorIsaacSim().update() # App.update()

        # Termination condition
        terminated = self.is_terminated()
        truncated = self.current_step >= self.max_steps
        
        observation = {
            "gridmap": self._get_local_gridmap(), 
            "state": self._get_drone_state() 
        }

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = self._computeReward()

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )


    ################################################################################

    def is_terminated(self):
        """Terminates the environment.
        """
        offset = 4.0
        min_bounds = self.sense_gridmap.realmap_bounds[0] - offset
        max_bounds = self.sense_gridmap.realmap_bounds[1] + offset
        self.is_out_of_bounds = torch.any(self.state.position  < min_bounds) or torch.any(self.state.position  > max_bounds)
        self.distance = torch.norm(self.state.position  - self.target_position)
        self.is_flipover = False

        # Judge whether the drone has reached the boundary
        if self.is_out_of_bounds:
            print(f"Out of bounds!")
            return True

        # Judge whether the drone has reached the target
        if self.distance < CONTROL_PARAMS["target_radius"]:
            print(f"Reach the target with distance: {self.distance}")
            return True
        
        # Judge whether the drone has flipped over and fallen to the ground
        angle_threshold = torch.tensor(np.pi / 2, dtype=torch.float32)
        roll, pitch, _ = self.state.orient
        if torch.abs(roll) > angle_threshold or torch.abs(pitch) > angle_threshold:
            self.is_flipover = True
            # print(f"Drone flips over with roll: {roll} and pitch: {pitch}")
            return True

        return False

    ################################################################################

    def close(self):
        """Terminates the environment.
        """
        pass

    ################################################################################
    
    def _actionSpace(self):
        """Returns the action space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    

    ################################################################################
    def _get_local_gridmap(self):
        """Returns the local map from the drone sensors.
        """
        self.state = self.quadrotor.state
        center = self.state.position
        local_map_shape = self.grid_map_dim[1:]

        local_map = self.sense_gridmap.get_local_map(center, local_map_shape)
        return local_map

    def _get_drone_state(self):
        """Returns the relative state of drone between the current and target.
        """
        # Update state
        self.state = self.quadrotor.state

        relative_state = torch.cat((self.target_position-self.state.position, 
                                 self.state.linear_velocity)).numpy()
        
        return relative_state
    
    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Must be implemented in a subclass.

        """
        
        self.success_reward = 10.0
        self.boundary_penalty = -20.0
        self.flipover_penalty = -20.0
        self.alpha = 0.1
        self.beta = 0.01

        speed = torch.norm(self.state.linear_velocity)

        if self.distance < CONTROL_PARAMS["target_radius"]:
            return self.success_reward
        elif self.is_out_of_bounds:
            return self.boundary_penalty
        elif self.is_flipover:
            return self.flipover_penalty
        else:
            # reward = -self.distance - self.alpha * speed - self.beta * self.current_step
            if self.last_distance is not None:
                if self.last_distance > self.distance: # 靠近goal
                    reward = 2*(self.last_distance - self.distance)
                else: # 离的更远带来的惩罚更大
                    reward = -4*(self.distance - self.last_distance)
                self.last_distance = self.distance
            else:
                self.last_distance = self.distance
                reward = torch.tensor(0.0)
            return reward.item()

# 在原有PerceptDroneNav类之后添加新的子类
class PerceptDroneNavSplitStep(PerceptDroneNav):
    """继承自PerceptDroneNav的新类，拆分step函数"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_target = None

    def step1(self, action):
        """执行物理模拟前的准备工作"""
        # 更新步数计数
        self.current_step += 1
        self.state = self.quadrotor.state

        # 将动作转换为轨迹点（原step函数前半部分）
        radius = 0.2
        x, y, z = spherical_to_cartesian(action, radius)
        x0 = self.state.position[0].item()
        y0 = self.state.position[1].item()
        z0 = self.state.position[2].item()
        x += x0
        y += y0
        z += z0
        psi = np.arctan2(y - y0, x - x0)
        psi += self.state.angular_velocity[2].item()
        psi = (psi + np.pi) % (2 * np.pi) - np.pi

        # 更新当前目标, 用于step()
        self.current_target = [x, y, z, psi]
        self.quadrotor.update_trajectory([self.current_target])
        # for _ in range(50):
        #     self.quadrotor.update_trajectory([self.current_target])
        #     QuadrotorIsaacSim().update()

    def step(self, action):
        """执行物理模拟和后续处理（原step函数后半部分）"""
        # 执行物理模拟（原循环中的App.update()部分）
        # TODO: 能不能不要循环update_trajectory(),
        # 既然每次给他的都一样，那没必要循环50次；
        # 在step1()中更新这个trajectory，然后就可以将QuadrotorIsaacSim().update()就可以在外面vec_env中做了

            # QuadrotorIsaacSim().update()

        # 终止条件判断和奖励计算（原step函数后半部分）
        terminated = self.is_terminated()
        truncated = self.current_step >= self.max_steps
        
        observation = {
            "gridmap": self._get_local_gridmap(), 
            "state": self._get_drone_state() 
        }

        reward = self._computeReward()
        info = {}

        return observation, reward, terminated, truncated, info


# 在PerceptDroneNav类之后添加
class DiscretePerceptDroneNav(PerceptDroneNav):
    """离散动作空间版本，继承原有环境"""
    
    def __init__(self, n_psi_bins=16, n_theta_bins=5, **kwargs):
        super().__init__(**kwargs)
        
        # 定义离散动作空间（ψ分n_psi_bins档，θ分n_theta_bins档）
        self.action_space = spaces.MultiDiscrete([n_psi_bins, n_theta_bins])
        self._n_psi = n_psi_bins
        self._n_theta = n_theta_bins
        
    def _discrete_to_continuous(self, action):
        """将离散动作转换为原环境的连续值"""
        psi = (action[0] / (self._n_psi - 1)) * 2 - 1  # [-1, 1]
        theta = (action[1] / (self._n_theta - 1)) * 2 - 1  # [-1, 1]
        return np.array([psi, theta], dtype=np.float32)
    
    def step(self, action):
        # 转换动作后调用父类逻辑
        continuous_action = self._discrete_to_continuous(action)
        return super().step(continuous_action)