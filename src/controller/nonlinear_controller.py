#!/usr/bin/env python
"""
| File: nonlinear_controller.py
| Author: Marcelo Jacinto and Joao Pinto (marcelo.jacinto@tecnico.ulisboa.pt, joao.s.pinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to use the control backends API to create a custom controller 
for the vehicle from scratch and use it to perform a simulation, without using PX4 nor ROS. In this controller, we
provide a quick way of following a given trajectory specified in csv files or track an hard-coded trajectory based
on exponentials! NOTE: This is just an example, to demonstrate the potential of the API. A much more flexible solution
can be achieved
"""

# Imports to be able to log to the terminal with fancy colors
import carb
import time
# Auxiliary scipy and numpy modules
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from omni.isaac.dynamic_control import _dynamic_control


class NonlinearController():
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
        default_env_path: str=None, 
        results_file: str=None,
        Kp=[1.0, 1.0, 1.0],
        Kd=[0.85, 0.85, 0.85],
        Ki=[0.15, 0.15, 0.15],
        Kr=[0.35, 0.35, 0.35],
        Kw=[0.05, 0.05, 0.05]):

        self.default_env_path = default_env_path
        self._vehicle_dc_interface = _dynamic_control.acquire_dynamic_control_interface()
        self._num_rotors = 4  # Four propellers

        # The current rotor references [rad/s]
        self.input_ref = [0.0, 0.0, 0.0, 0.0]

        # The current state of the vehicle expressed in the inertial frame (in ENU)
        self.p = torch.zeros(3)                          # The vehicle position
        self.R = Rotation.identity()                     # The vehicle attitude (stays as Rotation object)
        self.w = torch.zeros(3)                          # The angular velocity of the vehicle
        self.v = torch.zeros(3)                          # The linear velocity of the vehicle in the inertial frame
        self.a = torch.zeros(3)                          # The linear acceleration of the vehicle in the inertial frame
        self.int = torch.tensor([0.0, 0.0, 0.0])         # The integral of position error

        # Define the control gains matrix for the outer-loop
        self.Kp = torch.diag(torch.tensor(Kp))
        self.Kd = torch.diag(torch.tensor(Kd))
        self.Ki = torch.diag(torch.tensor(Ki))
        self.Kr = torch.diag(torch.tensor(Kr))
        self.Kw = torch.diag(torch.tensor(Kw))

        # Set the initial time for starting when using the built-in trajectory (the time is also used in this case
        # as the parametric value)
        self.total_time = 0.0
        # Signal that we will not used a received trajectory
        self.trajectory = None

        # Auxiliar variable, so that we only start sending motor commands once we get the state of the vehicle
        self.received_first_state = False

        # Lists used for analysing performance statistics
        self.results_files = results_file

    def start(self,gravity=9.81):
        """
        Reset the control and trajectory index
        """
        # Define the dynamic parameters for the vehicle
        self.m = self.get_mass()       # Mass in Kg
        self.g = gravity               # The gravity acceleration ms^-2

        self.reset_statistics()

    def stop(self):
        """
        Stopping the controller. Saving the statistics data for plotting later
        Here, the variables loaded into the statistics dictionary are all 'list' type
        """
        # Check if we should save the statistics to some file or not
        if self.results_files is None:
            return
        
        statistics = {}
        statistics["time"] = np.array(self.time_vector)
        statistics["p"] = np.vstack(self.position_over_time)
        statistics["desired_p"] = np.vstack(self.desired_position_over_time)
        statistics["ep"] = np.vstack(self.position_error_over_time)
        statistics["ev"] = np.vstack(self.velocity_error_over_time)
        statistics["er"] = np.vstack(self.attitude_error_over_time)
        statistics["ew"] = np.vstack(self.attitude_rate_error_over_time)
        np.savez(self.results_files, **statistics)
        carb.log_warn("Statistics saved to: " + self.results_files)

        self.reset_statistics()

    def update_state(self, state):
        """
        Method that updates the current state of the vehicle. This is a callback that is called at every physics step

        Args:
            state (State): The current state of the vehicle.
        """
        self.p = state["position"]
        self.R = Rotation.from_euler('xyz', state["attitude"])
        self.w = state["angular_velocity"]
        self.v = state["linear_velocity"]

        self.received_first_state = True

    def update_trajectory(self, traj):
        """
        Method that updates the current target trajectory point of the vehicle. This is a callback that is called at every physics step

        Args:
            traj: The current target trajectory point of the vehicle. An example is given:
                traj = {
                            "position": [x, y, z],       # the target positions [m]
                            "velocity": [vx, vy, vz],    # velocity [m/s]
                            "acceleration": [ax, ay, az],    # accelerations [m/s^2]
                            "jerk": [jx, jy, jz],    # jerk [m/s^3]
                            "yaw_angle": yaw,           # yaw-angle [rad]
                            "yaw_rate": yaw_rate  # yaw-rate [rad/s]
                        }
        """
        self.p_ref = torch.tensor(traj["position"], dtype=torch.float)          # 3-element tensor for position
        self.v_ref = torch.tensor(traj["velocity"], dtype=torch.float)          # 3-element tensor for velocity
        self.a_ref = torch.tensor(traj["acceleration"], dtype=torch.float)      # 3-element tensor for acceleration
        self.j_ref = torch.tensor(traj["jerk"], dtype=torch.float)              # 3-element tensor for jerk
        self.yaw_ref = torch.tensor(traj["yaw_angle"], dtype=torch.float)       # scalar tensor for yaw
        self.yaw_rate_ref = torch.tensor(traj["yaw_rate"], dtype=torch.float)   # scalar tensor for yaw rate

        return self.p_ref, self.v_ref, self.a_ref, self.j_ref, self.yaw_ref, self.yaw_rate_ref

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

        # -------------------------------------------------
        # Update the references for the controller to track
        # -------------------------------------------------
        self.total_time += dt

        # Update using an external trajectory
        # p_ref = torch.zeros(3)
        # v_ref = torch.zeros(3)
        # a_ref = torch.zeros(3)
        # j_ref = torch.zeros(3)
        # yaw_ref = torch.tensor(0, dtype=torch.float)
        # yaw_rate_ref = torch.tensor(0, dtype=torch.float)
        p_ref, v_ref, a_ref, j_ref, yaw_ref, yaw_rate_ref = self.p_ref, self.v_ref, self.a_ref, self.j_ref, self.yaw_ref, self.yaw_rate_ref

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
        print("ep: ", ep)
        print("ev: ", ev)
        
        # Compute F_des term
        F_des = -(self.Kp @ ep) - (self.Kd @ ev) - (self.Ki @ ei) + torch.tensor([0.0, 0.0, self.m * self.g]) + (self.m * a_ref)
        print("F_des: ", F_des)
        # Get the current axis Z_B (given by the last column of the rotation matrix)
        Z_B = torch.tensor(self.R.as_matrix()[:, 2], dtype=torch.float)
        print("Z_B: ", Z_B)
        # Get the desired total thrust in Z_B direction (u_1)
        u_1 = torch.dot(F_des, Z_B.squeeze())
        
        # Compute the desired body-frame axis Z_b
        Z_b_des = F_des / torch.norm(F_des)
        print("Z_b_des: ", Z_b_des)
        # Compute X_C_des
        X_c_des = torch.tensor([torch.cos(yaw_ref), torch.sin(yaw_ref), 0.0])
        print("X_c_des: ", X_c_des)
        # Compute Y_b_des
        Z_b_cross_X_c = torch.linalg.cross(Z_b_des, X_c_des)
        print("Z_b_cross_X_c: ", Z_b_cross_X_c)
        Y_b_des = Z_b_cross_X_c / torch.norm(Z_b_cross_X_c)
        print("Y_b_des: ", Y_b_des)
        # Compute X_b_des
        X_b_des = torch.cross(Y_b_des, Z_b_des)
        print("X_b_des: ", X_b_des)
        # Compute the desired rotation R_des = [X_b_des | Y_b_des | Z_b_des]
        R_des = torch.column_stack((X_b_des, Y_b_des, Z_b_des))
        R = torch.tensor(self.R.as_matrix().squeeze(0), dtype=torch.float)
        
        # Compute the rotation error
        e_R = 0.5 * self.vee((R_des.T @ R) - (R.T @ R_des))
        if e_R.dim() == 2 and e_R.shape[0] == 1:
            e_R = e_R.squeeze(0)
        print("e_R: ", e_R)
        # Compute an approximation of the current vehicle acceleration in the inertial frame (since we cannot measure it directly)
        self.a = (u_1 * Z_B) / self.m - torch.tensor([0.0, 0.0, self.g])
        print("self.a: ", self.a)
        # Compute the desired angular velocity by projecting the angular velocity in the Xb-Yb plane
        hw = (self.m / u_1) * (j_ref - torch.dot(Z_b_des, j_ref) * Z_b_des)
        print("hw: ", hw)
        # desired angular velocity
        w_des = torch.tensor([-torch.dot(hw, Y_b_des), torch.dot(hw, X_b_des), yaw_rate_ref * Z_b_des[2]], dtype=torch.float)
        print("w_des: ", w_des)
        # Compute the angular velocity error
        e_w = self.w - w_des
        if e_w.dim() == 2 and e_w.shape[0] == 1:
            e_w = e_w.squeeze(0)
        print("e_w: ", e_w)
        # Compute the torques to apply on the rigid body
        tau = -(self.Kr @ e_R) - (self.Kw @ e_w)
        print("controller force: ", u_1, tau)
        
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

        return u_1, tau

    def get_dc_interface(self):
        """ Get interface of Dynamic Control """
        if self._vehicle_dc_interface is None:
            self._vehicle_dc_interface = _dynamic_control.acquire_dynamic_control_interface()
        return self._vehicle_dc_interface

    def get_mass(self):
        """ Get the overall mass of Crazyflie robot """
        self.drone_stage_prefix = self.default_env_path + "/Crazyflie"
        # get mass of main body
        rb_body = self._vehicle_dc_interface.get_rigid_body(self.drone_stage_prefix + "/body")
        body_properties = self._vehicle_dc_interface.get_rigid_body_properties(rb_body)
        mass = body_properties.mass

        # get mass of four propellers
        rotors = [self._vehicle_dc_interface.get_rigid_body(self.drone_stage_prefix + "/m" + str(i+1) + "_prop") for i in range(self._num_rotors)]
        for rotor in rotors:
            properties = self._vehicle_dc_interface.get_rigid_body_properties(rotor)
            mass += properties.mass
        print("Mass: ", mass)
        
        return mass
    
    def get_rotor_relative_positions(self):
        # Get the rigid body handle of the drone in body frame
        rb_body = self._vehicle_dc_interface.get_rigid_body(self.drone_stage_prefix + "/body")
        
        # Get the rigid body handles for four propellers
        rotors = [self._vehicle_dc_interface.get_rigid_body(self.drone_stage_prefix + "/m" + str(i+1) + "_prop") for i in range(self._num_rotors)]
        
        # Get the relative positions of the four propellers with respect to the body (ignoring rotation)
        relative_poses = self._vehicle_dc_interface.get_relative_body_poses(rb_body, rotors)
        
        # Calculate location information
        relative_positions = []
        for i, pose in enumerate(relative_poses):
            position = pose.p  # p is the translation
            relative_position = torch.tensor([position.x, position.y, position.z], dtype=torch.float32)
            relative_positions.append(relative_position)
            # print(f"Relative position of rotor {i} to body frame: {relative_position}")
        
        return relative_positions

    def force_and_torques_to_thrust(self, force=None, torque=None):
        """
        Auxiliar method used to get the target angular velocities for each rotor, given the total desired thrust [N] and
        torque [Nm] to be applied in the multirotor's body frame.

        Note: This method assumes a quadratic thrust curve. This method will be improved in a future update,
        and a general thrust allocation scheme will be adopted. For now, it is made to work with multirotors directly.

        Args:
            force (np.ndarray): A vector of the force to be applied in the body frame of the vehicle [N]
            torque (np.ndarray): A vector of the torque to be applied in the body frame of the vehicle [Nm]

        Returns:
            list: A list of angular velocities [rad/s] to apply in reach rotor to accomplish suchs forces and torques
        """

        # Integrate input variables
        # force_torque = torch.cat([torch.from_numpy(force), torch.from_numpy(torque)])
        force_torque = torch.cat([force.unsqueeze(0), torque])

        # Get the relative positions of the four propellers with respect to the body
        self.drone_stage_prefix = self.default_env_path + "/Crazyflie"
        rel_pos = self.get_rotor_relative_positions()

        # Configuration parameters of propellers
        cT = 1
        cM = 1e-6
        nT = -1*cT

        # Construct allocation matrix A
        A = torch.zeros(self._num_rotors, force_torque.numel())
        A = torch.tensor([
            [cT, cT, cT, cT],
            [rel_pos[0][1]*cT, rel_pos[1][1]*cT, rel_pos[2][1]*cT, rel_pos[3][1]*cT],
            [rel_pos[0][0]*nT, rel_pos[1][0]*nT, rel_pos[2][0]*nT, rel_pos[3][0]*nT],
            [-cM, cM, -cM, cM]
        ], dtype=torch.float32)

        print("Allocation matrix: ", A)
        # Compute the target thrust of propellers
        rotor_thrusts = torch.linalg.solve(A, force_torque)

        print("rotor_thrusts: ", rotor_thrusts)

        return rotor_thrusts
    
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