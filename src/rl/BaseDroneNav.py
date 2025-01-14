#!/usr/bin/env python
import numpy as np
import torch
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
# Isaac Sim APIs
from envs.isaacgym_env import QuadrotorIsaacSim
from configs.configs import ROBOT_PARAMS, CONTROL_PARAMS
from controller.nonlinear_controller import NonlinearController
from map.sense_gridmap import SenseGridMap
from sensors.lidar import RotatingLidar
from robots.quadrotor import Quadrotor

# gym APIs
import gymnasium as gym
from gymnasium import spaces

class BaseDroneNav(gym.Env):
    """Base class for "drone navigation" Gym environments."""
    # The isaac sim simulation environment has already been created
    # metadata = {"render_modes": ["human"], "render_fps": 30}

    ################################################################################

    def __init__(self,
                 record=False,
                 output_folder='results'
                 ):
        
        """Initialization of a generic aviary environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.

        """
        super(BaseDroneNav, self).__init__()

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
        # The action will be normalized x,y,z,psi coordinates
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # The observation will be the coordinate of the agent
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6,), 
            dtype=np.float32
        )

        # Configurations
        self.target_position = torch.tensor(CONTROL_PARAMS["target_position"], dtype=torch.float32)
        self.max_steps = 200
        self.current_step = 0


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
        observation = torch.cat((self.target_position-self.quadrotor.state.position, 
                                 self.quadrotor.state.linear_velocity)).numpy()
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
        scaling_factor_position = 0.2
        x, y, z = action[:3] * scaling_factor_position  
        x += self.state.position[0].item()  # Add current x position
        y += self.state.position[1].item()  # Add current y position
        z += self.state.position[2].item()  # Add current z position

        psi = action[3] * np.pi/20
        psi += self.state.angular_velocity[2].item()  # Add current psi angle

        # Apply new trajectory point to quadrotor
        for i in range(30):
            self.quadrotor.update_trajectory([[x, y, z, psi]])
            QuadrotorIsaacSim().update() # App.update()

        # Termination condition
        terminated = self.is_terminated()
        truncated = self.current_step >= self.max_steps

        observation = torch.cat((self.target_position-self.state.position, 
                                 self.state.linear_velocity)).numpy()
        
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
            print(f"Drone flips over with roll: {roll} and pitch: {pitch}")
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

    def _computeReward(self):
        """Computes the current reward value(s).

        Must be implemented in a subclass.

        """
        
        self.success_reward = 100.0
        self.boundary_penalty = -50.0
        self.alpha = 0.1
        self.beta = 0.01

        speed = torch.norm(self.state.linear_velocity)

        if self.distance < CONTROL_PARAMS["target_radius"]:
            return self.success_reward
        elif self.is_out_of_bounds:
            return -self.boundary_penalty
        else:
            reward = -self.distance - self.alpha * speed - self.beta * self.current_step
            return reward.item()
        

if __name__ == "__main__":
    QIS = QuadrotorIsaacSim()

    from stable_baselines3.common.env_checker import check_env

    env = BaseDroneNav()
    check_env(env, warn=True)