#!/usr/bin/env python
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

# Imports to be able to log to the terminal with fancy colors
import carb

# Auxiliary scipy and numpy modules
import torch
from scipy.spatial.transform import Rotation

# 
from omni.isaac.dynamic_control import _dynamic_control
from utils.state import State
from utils.task_util import traj_tensor
from controller.min_snap_traj import SnapTrajectory
from controller.backend import Backend
from configs.configs import CONTROL_PARAMS


class NonlinearController(Backend):
    """A nonlinear controller class. It implements a nonlinear controller that allows a vehicle to track
    aggressive trajectories. This controlers is well described in the papers
    
    [1] J. Pinto, B. J. Guerreiro and R. Cunha, "Planning Parcel Relay Manoeuvres for Quadrotors," 
    2021 International Conference on Unmanned Aircraft Systems (ICUAS), Athens, Greece, 2021, 
    pp. 137-145, doi: 10.1109/ICUAS51884.2021.9476757.
    [2] D. Mellinger and V. Kumar, "Minimum snap trajectory generation and control for quadrotors," 
    2011 IEEE International Conference on Robotics and Automation, Shanghai, China, 2011, 
    pp. 2520-2525, doi: 10.1109/ICRA.2011.5980409.
    """

    def __init__(self, 
        stage_prefix: str=None,
        results_file: str=None,
        Kp=[10.0, 10.0, 10.0],
        Kd=[8.5, 8.5, 8.5],
        Ki=[1.50, 1.50, 1.50],
        Kr=[3.5, 3.5, 3.5],
        Kw=[0.5, 0.5, 0.5]):

        # Define the dynamic parameters for the vehicle
        self.m = CONTROL_PARAMS['mass']      # Mass in Kg
        self.g = CONTROL_PARAMS['gravity']   # The gravity acceleration ms^-2
        self._num_rotors = CONTROL_PARAMS['num_rotors']

        # Handle input
        self._stage_prefix = stage_prefix
        self.results_files = results_file # Lists used for analysing performance statistics

        # The current rotor references [rad/s]
        self.input_ref = torch.tensor([0.0 for i in range(self._num_rotors)], dtype=torch.float32)

        # The current state of the vehicle expressed in the inertial frame (in ENU)
        self.p = torch.zeros(3)                          # The vehicle position
        self.R: Rotation = Rotation.identity()           # The vehicle attitude (stays as Rotation object)
        self.w = torch.zeros(3)                          # The angular velocity of the vehicle
        self.v = torch.zeros(3)                          # The linear velocity of the vehicle in the inertial frame
        self.a = torch.zeros(3)                          # The linear acceleration of the vehicle in the inertial frame
        self.orient = torch.zeros(3)

        self.int = torch.tensor([0.0, 0.0, 0.0])         # The integral of position error

        # Define the control gains matrix for the outer-loop
        self.Kp = torch.diag(torch.tensor(Kp))
        self.Kd = torch.diag(torch.tensor(Kd))
        self.Ki = torch.diag(torch.tensor(Ki))
        self.Kr = torch.diag(torch.tensor(Kr))
        self.Kw = torch.diag(torch.tensor(Kw))

        self._vehicle_dc_interface = None

        # Set the initial time for starting when using the built-in trajectory (the time is also used in this case
        # as the parametric value)
        self.total_time = 0.0
        self.index = 0
        # Signal that we will not used a received trajectory
        self.trajectory = None
        self.traj_generator = SnapTrajectory(7) # self.traj_generator.traj(waypoints)

        # Auxiliar variable, so that we only start sending motor commands once we get the state of the vehicle
        self.received_first_state = False

        # Lists used for analysing performance statistics
        self.results_files = results_file
        self.reset_statistics()

    def start(self):
        """
        Reset the control and trajectory index
        """
        self.reset_statistics()

    def stop(self):
        """
        Stopping the controller. Saving the statistics data for plotting later
        Here, the variables loaded into the statistics dictionary are all 'list' type
        """
        # Check if we should save the statistics to some file or not
        if self.results_files is None:
            return
        
        # Convert lists to torch tensors
        statistics = {}
        statistics["time"] = torch.tensor(self.time_vector, dtype=torch.float32)
        statistics["p"] = torch.stack(self.position_over_time)
        statistics["desired_p"] = torch.stack(self.desired_position_over_time)
        statistics["ep"] = torch.stack(self.position_error_over_time)
        statistics["ev"] = torch.stack(self.velocity_error_over_time)
        statistics["er"] = torch.stack(self.attitude_error_over_time)
        statistics["ew"] = torch.stack(self.attitude_rate_error_over_time)
        torch.save(statistics, self.results_files)

        carb.log_warn("Statistics saved to: " + self.results_files)

        self.reset_statistics()

    def update_state(self, state: State):
        """
        Method that updates the current state of the vehicle. This is a callback that is called at every physics step

        Args:
            state (State): The current state of the vehicle.
        """
        self.p = state.position
        self.R = state.R
        self.orient = state.orient # as_euler('ZYX', degrees=True)
        self.v = state.linear_velocity
        self.w = state.angular_velocity

        self.received_first_state = True
    
    def update_trajectory(self, points):
        """
        Generate waypoints based on current state and path points.
        Args:
            points (list): List of path points, each as [x, y, z, yaw].
        Returns:
            list: Waypoints.
        """
        # integrate the current state with waypoints
        waypoints = []
        t0 = 0.0
        x0, y0, z0 = self.p.tolist()
        psi0 = self.orient[2].item()
        dx0, dy0, dz0 = self.v .tolist()
        dpsi0 = self.w[2].item()
        waypoints.extend([
        [x0, dx0],  # Initial x and its derivatives
        [y0, dy0],  # Initial y and its derivatives
        [z0, dz0],  # Initial z and its derivatives
        [psi0, dpsi0],         # Initial yaw and its derivatives
        [t0]])

        # Add each point in the path
        dt = CONTROL_PARAMS['control_cycle']
        for i, point in enumerate(points):
            x, y, z, psi = point
            waypoints.extend([[x], [y], [z], [psi], [t0 + (i+1)*dt]])

        # 
        self.traj_generator.traj(waypoints)
        self.trajectory = self.traj_generator.output_traj_points()
        self.max_index = len(self.trajectory)
        self.index = 0
        self.traj_time = 0.0
        self.time_ref = 0.0
        # carb.log_warn(f"trajectory is: {self.trajectory}")

    def input_reference(self):
        """
        Method that is used to return the latest target angular velocities to be applied to the vehicle

        Returns:
            A list with the target angular velocities for each individual rotor of the vehicle
        """
        return self.input_ref

    def update(self, dt: float):
        """
        Method that implements the nonlinear control law and updates the target angular velocities for each rotor. 
        This method will be called by the simulation on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        if self.received_first_state == False:
            return
        self.total_time += dt
        # -------------------------------------------------
        # Update the references for the controller to track
        # -------------------------------------------------
        
        self.traj_time += dt
        if self.index < self.max_index - 1 and self.traj_time >= self.time_ref:
            self.index += 1
        # carb.log_warn(f"Trajectory index is: {self.index}")
        # Update using an external trajectory
        if self.trajectory is not None:
            p_ref, v_ref, a_ref, j_ref, yaw_ref, yaw_rate_ref, self.time_ref = traj_tensor(self.index, self.trajectory)
        else:
            p_ref = self.p
            v_ref = self.v
            a_ref = torch.zeros(3) 
            j_ref = torch.zeros(3)
            yaw_ref = self.orient[2].item()
            yaw_rate_ref = self.w[2].item()
            self.time_ref = 0.0
        # carb.log_warn(f"Reference point is: {p_ref}, {yaw_ref}, {v_ref}")
        # -------------------------------------------------
        # Start the controller implementation
        # -------------------------------------------------

        # Compute the tracking errors
        ep = self.p - p_ref
        if ep.dim() == 2 and ep.shape[0] == 1:
            ep = ep.squeeze(0)
        ev = self.v - v_ref
        if ev.dim() == 2 and ev.shape[0] == 1:
            ev = ev.squeeze(0)
        self.int = self.int + (ep * dt)
        ei = self.int
        
        # Compute F_des term
        F_des = -(self.Kp @ ep) - (self.Kd @ ev) - (self.Ki @ ei) + torch.tensor([0.0, 0.0, self.m * self.g]) + (self.m * a_ref)
        # Get the current axis Z_B (given by the last column of the rotation matrix)
        Z_B = torch.tensor(self.R.as_matrix()[:, 2], dtype=torch.float)
        # Get the desired total thrust in Z_B direction (u_1)
        u_1 = torch.dot(F_des, Z_B.squeeze())
        
        # Compute the desired body-frame axis Z_b
        Z_b_des = F_des / torch.norm(F_des)
        # Compute X_C_des
        X_c_des = torch.tensor([torch.cos(yaw_ref), torch.sin(yaw_ref), 0.0])
        # Compute Y_b_des
        Z_b_cross_X_c = torch.linalg.cross(Z_b_des, X_c_des)
        Y_b_des = Z_b_cross_X_c / torch.norm(Z_b_cross_X_c)
        # Compute X_b_des
        X_b_des = torch.cross(Y_b_des, Z_b_des, dim=0)
        # Compute the desired rotation R_des = [X_b_des | Y_b_des | Z_b_des]
        R_des = torch.column_stack((X_b_des, Y_b_des, Z_b_des))
        R = torch.tensor(self.R.as_matrix(), dtype=torch.float)
        
        # Compute the rotation error
        e_R = 0.5 * self.vee((R_des.T @ R) - (R.T @ R_des))
        if e_R.dim() == 2 and e_R.shape[0] == 1:
            e_R = e_R.squeeze(0)
        # Compute an approximation of the current vehicle acceleration in the inertial frame (since we cannot measure it directly)
        self.a = (u_1 * Z_B) / self.m - torch.tensor([0.0, 0.0, self.g])
        # Compute the desired angular velocity by projecting the angular velocity in the Xb-Yb plane
        hw = (self.m / u_1) * (j_ref - torch.dot(Z_b_des, j_ref) * Z_b_des)
        # desired angular velocity
        w_des = torch.tensor([-torch.dot(hw, Y_b_des), torch.dot(hw, X_b_des), yaw_rate_ref * Z_b_des[2]], dtype=torch.float)
        # Compute the angular velocity error
        e_w = self.w - w_des
        if e_w.dim() == 2 and e_w.shape[0] == 1:
            e_w = e_w.squeeze(0)
        # Compute the torques to apply on the rigid body
        tau = -(self.Kr @ e_R) - (self.Kw @ e_w)
        
        if self.vehicle:
            self.input_ref = self.vehicle.force_and_torques_to_velocities(u_1, tau)
        # carb.log_warn(f"The input_reference is: {self.input_ref}")
        # ----------------------------
        # Statistics to save for later
        # ----------------------------
        self.time_vector.append(self.total_time)
        self.position_over_time.append(self.p)
        self.desired_position_over_time.append(p_ref)
        self.position_error_over_time.append(ep)
        self.velocity_error_over_time.append(ev)
        self.attitude_error_over_time.append(e_R)
        self.attitude_rate_error_over_time.append(e_w)

    def get_dc_interface(self):
        """ Get interface of Dynamic Control """
        if self._vehicle_dc_interface is None:
            self._vehicle_dc_interface = _dynamic_control.acquire_dynamic_control_interface()
        return self._vehicle_dc_interface

    def get_mass(self):
        """ Get the overall mass of robot """
        # get mass of main body
        rb_body = self.get_dc_interface().get_rigid_body(self._stage_prefix + "/body")
        body_properties = self.get_dc_interface().get_rigid_body_properties(rb_body)
        mass = body_properties.mass

        # get mass of four propellers
        rotors = [self.get_dc_interface().get_rigid_body(self.drone_stage_prefix + "/rotor" + str(i)) for i in range(self._num_rotors)]
        for rotor in rotors:
            properties = self.get_dc_interface().get_rigid_body_properties(rotor)
            mass += properties.mass
        # print("Mass: ", mass)
        
        return mass
    
    @staticmethod
    def vee(S):
        """Auxiliary function that computes the 'v' map which takes elements from so(3) to R^3.

        Args:
            S (torch.Tensor): A matrix in so(3)
        """
        return torch.tensor([-S[1, 2], S[0, 2], -S[0, 1]], dtype=S.dtype, device=S.device)

    def reset_statistics(self):
        # If we received an external trajectory, reset the time to 0.0
        self.total_time = 0.0

        # Reset the lists used for analysing performance statistics
        self.time_vector = []
        self.desired_position_over_time = []
        self.position_over_time = []
        self.position_error_over_time = []
        self.velocity_error_over_time = []
        self.attitude_error_over_time = []
        self.attitude_rate_error_over_time = []

    def reset(self):
        # The current rotor references [rad/s]
        self.input_ref = torch.tensor([0.0 for i in range(self._num_rotors)], dtype=torch.float32)

        # The current state of the vehicle expressed in the inertial frame (in ENU)
        self.int = torch.tensor([0.0, 0.0, 0.0])         # The integral of position error
        
        self.update_state(self.vehicle._state)
        # Set the initial time for starting when using the built-in trajectory (the time is also used in this case
        # as the parametric value)
        self.index = 0
        # Signal that we will not used a received trajectory
        self.trajectory = None

        # Auxiliar variable, so that we only start sending motor commands once we get the state of the vehicle
        self.received_first_state = False

        # Lists used for analysing performance statistics
        self.reset_statistics()

        # Reset the trahectory generator
        self.traj_generator.reset()