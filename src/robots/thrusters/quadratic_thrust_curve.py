#
import sys
sys.path.append("../..")

import torch
from utils.state import State
from configs.configs import CONTROL_PARAMS

class QuadraticThrustCurve():
    """Class that implements the dynamics of rotors that can be described by a quadratic thrust curve
    """
    def __init__(self, config={}):
        """_summary_

        Args:
            config (dict): A Dictionary that contains all the parameters for configuring the QuadraticThrustCurve - it can be empty or only have some of the parameters used by the QuadraticThrustCurve.
        
        Examples:
            The dictionary default parameters are

            >>> {"num_rotors": 4,
            >>>  "rotor_constant": [5.84e-6, 5.84e-6, 5.84e-6, 5.84e-6],
            >>>  "rolling_moment_coefficient": [1e-6, 1e-6, 1e-6, 1e-6],
            >>>  "rot_dir": [-1, -1, 1, 1],
            >>>  "min_rotor_velocity": [0, 0, 0, 0],                      # rad/s
            >>>  "max_rotor_velocity": [1100, 1100, 1100, 1100],          # rad/s
            >>> }
        """

        # Get the total number of rotors to simulate
        self._num_rotors = config.get("num_rotors", CONTROL_PARAMS['num_rotors'])

        # The rotor constant used for computing the total thrust produced by the rotor: T = rotor_constant * omega^2
        self._rotor_constant = torch.tensor(config.get("rotor_constant", [8.54858e-6, 8.54858e-6, 8.54858e-6, 8.54858e-6]),  dtype=torch.float32)
        assert len(self._rotor_constant) == self._num_rotors

        # The rotor constant used for computing the total torque generated about the vehicle Z-axis
        self._rolling_moment_coefficient = torch.tensor(config.get("rolling_moment_coefficient", [1e-6, 1e-6, 1e-6, 1e-6]),  dtype=torch.float32)
        assert len(self._rolling_moment_coefficient) == self._num_rotors

        # Save the rotor direction of rotation
        self._rot_dir = torch.tensor(config.get("rot_dir", [-1, -1, 1, 1]),  dtype=torch.float32)
        assert len(self._rot_dir) == self._num_rotors

        # Values for the minimum and maximum rotor velocity in rad/s
        self.min_rotor_velocity = torch.tensor(config.get("min_rotor_velocity", [0, 0, 0, 0]),  dtype=torch.float32)
        assert len(self.min_rotor_velocity) == self._num_rotors

        self.max_rotor_velocity = torch.tensor(config.get("max_rotor_velocity", [1100, 1100, 1100, 1100]),  dtype=torch.float32)
        assert len(self.max_rotor_velocity) == self._num_rotors

        # The actual speed references to apply to the vehicle rotor joints
        self._input_reference = torch.tensor([0.0 for i in range(self._num_rotors)], dtype=torch.float32)

        # The actual velocity that each rotor is spinning at
        self._velocity = torch.tensor([0.0 for i in range(self._num_rotors)], dtype=torch.float32)

        # The actual force that each rotor is generating
        self._force = torch.tensor([0.0 for i in range(self._num_rotors)], dtype=torch.float32)

        # The actual rolling moment that is generated on the body frame of the vehicle
        self._rolling_moment = torch.tensor(0.0, dtype=torch.float32)

    def set_input_reference(self, input_reference):
        """
        Receives as input a tensor of target angular velocities of each rotor in rad/s
        """

        # The target angular velocity of the rotor
        self._input_reference = input_reference

    def update(self, state: State = None, dt: float = None):
        """
        Note: the state and dt variables are not used in this implementation, but left
        to add support to other rotor models where the total thrust is dependent on
        states such as vehicle linear velocity

        Args:
            state (State): The current state of the vehicle.
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        rolling_moment = torch.tensor(0.0, dtype=torch.float32)

        # Compute the actual force to apply to the rotors and the rolling moment contribution
        for i in range(self._num_rotors):
            # Set the actual velocity that each rotor is spinning at (instantaneous model - no delay introduced)
            # Only apply clipping of the input reference
            self._velocity[i] = torch.maximum(
                self.min_rotor_velocity[i], 
                torch.minimum(self._input_reference[i].clone().detach(), self.max_rotor_velocity[i])
            )

            # Set the force using a quadratic thrust curve
            self._force[i] = self._rotor_constant[i] * torch.pow(self._velocity[i], 2)

            # Compute the rolling moment coefficient
            rolling_moment += self._rolling_moment_coefficient[i] * torch.pow(self._velocity[i], 2) * self._rot_dir[i]

        # Update the rolling moment variable
        self._rolling_moment = rolling_moment
        
        # Return the forces and velocities on each rotor and total torque applied on the body frame
        return self._force, self._velocity, self._rolling_moment


    @property
    def force(self):
        """The force to apply to each rotor of the vehicle at any given time instant

        Returns:
            list: A list of forces (in Newton N) to apply to each rotor of the vehicle (on its Z-axis) at any given time instant
        """
        return self._force

    @property
    def velocity(self):
        """The velocity at which each rotor of the vehicle should be rotating at any given time instant

        Returns:
            list: A list of angular velocities (in rad/s) of each rotor (about its Z-axis) at any given time instant
        """
        return self._velocity

    @property
    def rolling_moment(self):
        """The total rolling moment being generated on the body frame of the vehicle by the rotating propellers

        Returns:
            float: The total rolling moment to apply to the vehicle body frame (Torque about the Z-axis) in Nm
        """
        return self._rolling_moment

    @property
    def rot_dir(self):
        """The direction of rotation of each rotor of the vehicle

        Returns:
            list(int): A list with the rotation direction of each rotor (-1 is counter-clockwise and 1 for clockwise)
        """
        return self._rot_dir

if __name__ == "__main__":
    # Initialize test inputs
    quad_thrust_curve = QuadraticThrustCurve()
    
    # show initial state
    print("Initial forces:", quad_thrust_curve.force)
    print("Initial velocities:", quad_thrust_curve.velocity)
    print("Initial rolling moment:", quad_thrust_curve.rolling_moment)

    # set input reference
    input_reference = torch.tensor([800, 900, 800, 1000], dtype=torch.float32)
    quad_thrust_curve.set_input_reference(input_reference)

    # update
    forces, velocities, rolling_moment = quad_thrust_curve.update()

    # show new state
    print("Updated forces:", forces)
    print("Updated velocities:", velocities)
    print("Updated rolling moment:", rolling_moment)
