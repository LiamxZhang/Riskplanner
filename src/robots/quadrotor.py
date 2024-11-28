# 
from typing import Optional

# Isaac sim APIs
import carb
import torch
from omni.isaac.core.utils.nucleus import get_assets_root_path

# Extension APIs
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from robots.vehicle import Vehicle
from robots.thrusters.quadratic_thrust_curve import QuadraticThrustCurve
from robots.dynamics.linear_drag import LinearDrag
from configs.configs import CONTROL_PARAMS
from envs.isaacgym_env import QuadrotorIsaacSim

# when self.prim_grid is needed, it can be accessed from 
# self.prim_grid = QuadrotorIsaacSim().prim_grid


def complete_usd_path(usd_file: str) -> str:
    """
    Complete the USD file path based on whether it is a local or remote path.
    Args:
        usd_file (str): The USD file path.
    Returns:
        str: The completed USD file path.
    """
    # Check if the path is already a remote Omniverse path
    if "omniverse://localhost" in usd_file:
        return usd_file

    # Otherwise, construct the local path
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        return ""

    return assets_root_path + usd_file

class Quadrotor(Vehicle):
    def __init__(
        self,
        stage_prefix: str,
        name: Optional[str] = "quadrotor",
        usd_path: Optional[str] = None,
        init_position: Optional[list] = None,
        init_orientation: Optional[list] = None,
        scale: Optional[list] = None,
        sensors: Optional[list] = [],
        graphical_sensors: Optional[list] = [],
        backends: Optional[list] = []
    ) -> None:
        
        # Set properties
        self._usd_path = complete_usd_path(usd_path)
        self._name = name
        
        # Setup the dynamics of the system - get the thrust curve of the vehicle from the configuration
        self._thrusters = QuadraticThrustCurve()
        self._drag = LinearDrag([0.50, 0.30, 0.0])

        # Initialize the Vehicle object
        super().__init__(
            stage_prefix=stage_prefix, 
            usd_path = self._usd_path, 
            init_pos=init_position, 
            init_orient=init_orientation, 
            scale=scale,
            sensors=sensors,
            graphical_sensors=graphical_sensors,
            backends=backends
            )

    def update(self, dt: float):
        """
        Method that computes and applies the forces to the vehicle in simulation. 
        This method must be implemented by a class that inherits this type. This callback
        is called on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Get the articulation root of the vehicle
        articulation = self.get_dc_interface().get_articulation(self._stage_prefix)

        # Get the desired angular velocities for each rotor
        if len(self._backends) != 0:
            desired_rotor_velocities = self._backends[0].input_reference()
        else:
            desired_rotor_velocities = torch.tensor([100.0 for i in range(self._thrusters._num_rotors)], dtype=torch.float32)

        # Input the desired rotor velocities in the thruster model
        self._thrusters.set_input_reference(desired_rotor_velocities)

        # Get the desired forces to apply to the vehicle
        forces_z, _, rolling_moment = self._thrusters.update(self._state, dt)
        # forces_z_str = ', '.join([f"{val:.4f}" for val in forces_z.tolist()]) 
        # carb.log_error(f"Forces on Z-axis (forces_z): : {forces_z_str}")
        # carb.log_error(f"Rolling moment: : {rolling_moment.item():.4f}")
        # Apply force to each rotor
        for i in range(CONTROL_PARAMS['num_rotors']):
            # Apply the force in Z on the rotor frame
            self.apply_force([0.0, 0.0, forces_z[i].item()], body_part="/rotor" + str(i))

            # Generate the rotating propeller visual effect
            self.handle_propeller_visual(i, forces_z[i].item(), articulation)
        
        # Apply the torque to the body frame of the vehicle that corresponds to the rolling moment
        self.apply_torque([0.0, 0.0, rolling_moment.item()], "/body")
        # Compute the total linear drag force to apply to the vehicle's body frame
        drag = self._drag.update(self._state, dt)
        self.apply_force(drag.tolist(), body_part="/body")

        # Call the update methods in backends
        for backend in self._backends:
            backend.update(dt)

    def update_trajectory(self, points):
        for backend in self._backends:
            backend.update_trajectory(points)

    def handle_propeller_visual(self, rotor_number, force: float, articulation):
        """
        Auxiliar method used to set the joint velocity of each rotor (for animation purposes) based on the 
        amount of force being applied on each joint

        Args:
            rotor_number (int): The number of the rotor to generate the rotation animation
            force (float): The force that is being applied on that rotor
            articulation (_type_): The articulation group the joints of the rotors belong to
        """

        # Rotate the joint to yield the visual of a rotor spinning (for animation purposes only)
        joint = self.get_dc_interface().find_articulation_dof(articulation, "joint" + str(rotor_number))

        # Spinning when armed but not applying force
        if 0.0 < force < 0.1:
            self.get_dc_interface().set_dof_velocity(joint, -5 * self._thrusters.rot_dir[rotor_number])
        # Spinning when armed and applying force
        elif 0.1 <= force:
            self.get_dc_interface().set_dof_velocity(joint, -100 * self._thrusters.rot_dir[rotor_number])
        # Not spinning
        else:
            self.get_dc_interface().set_dof_velocity(joint, 0)

    
    def force_and_torques_to_velocities(self, force: float, torque: torch.Tensor):
        """
        Compute the rotor angular velocities based on the input force and torques.
        
        Args:
            force (float): Total thrust force.
            torque (torch.Tensor): A tensor of shape (3,) representing torques [\tau_x, \tau_y, \tau_z].
        
        Returns:
            torch.Tensor: A tensor of rotor angular velocities in [rad/s].
        """
        # Get the body frame of the vehicle
        rb = self.get_dc_interface().get_rigid_body(self._stage_prefix + "/body")

        # Get the rotors of the vehicle
        rotors = [self.get_dc_interface().get_rigid_body(self._stage_prefix + "/rotor" + str(i)) for i in range(self._thrusters._num_rotors)]

        # Get the relative position of the rotors with respect to the body frame of the vehicle (ignoring the orientation for now)
        relative_poses = self.get_dc_interface().get_relative_body_poses(rb, rotors)

        # Define the allocation matrix
        aloc_matrix = torch.zeros((4, self._thrusters._num_rotors), dtype=torch.float32)

        # Define the first line of the matrix (T [N])
        aloc_matrix[0, :] = torch.tensor(self._thrusters._rotor_constant, dtype=torch.float32)

        # Define the second and third lines of the matrix (\tau_x [Nm] and \tau_y [Nm])
        aloc_matrix[1, :] = torch.tensor(
            [relative_poses[i].p[1] * self._thrusters._rotor_constant[i] for i in range(self._thrusters._num_rotors)],
            dtype=torch.float32
        )
        aloc_matrix[2, :] = torch.tensor(
            [-relative_poses[i].p[0] * self._thrusters._rotor_constant[i] for i in range(self._thrusters._num_rotors)],
            dtype=torch.float32
        )

        # Define the fourth line of the matrix (\tau_z [Nm])
        aloc_matrix[3, :] = torch.tensor(
            [self._thrusters._rolling_moment_coefficient[i] * self._thrusters._rot_dir[i] for i in range(self._thrusters._num_rotors)],
            dtype=torch.float32
        )

        # Compute the pseudo-inverse of the allocation matrix
        aloc_inv = torch.linalg.pinv(aloc_matrix)

        # Compute the target angular velocities (squared)
        squared_ang_vel = aloc_inv @ torch.tensor([force, torque[0], torque[1], torque[2]], dtype=torch.float32)

        # Ensure there are no negative values for the squared angular velocities
        squared_ang_vel = torch.clamp(squared_ang_vel, min=0.0)

        # ------------------------------------------------------------------------------------------------
        # Saturate the inputs while preserving their relation to each other, by performing a normalization
        # ------------------------------------------------------------------------------------------------
        max_thrust_vel_squared = torch.pow(self._thrusters.max_rotor_velocity[0], 2)
        max_val = torch.max(squared_ang_vel)

        if max_val >= max_thrust_vel_squared:
            normalize = max(max_val / max_thrust_vel_squared, 1.0)
            squared_ang_vel = squared_ang_vel / normalize

        # Compute the angular velocities for each rotor in [rad/s]
        ang_vel = torch.sqrt(squared_ang_vel)

        return ang_vel

if __name__ == "__main__":
    #
    pass