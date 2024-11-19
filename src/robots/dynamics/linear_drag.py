#
import sys
sys.path.append("../..")

import torch
from utils.state import State

class LinearDrag:
    """
    Class that implements linear drag computations affecting a rigid body.
    """

    def __init__(self, drag_coefficients=[0.0, 0.0, 0.0]):
        """
        Receives as input the drag coefficients of the vehicle as a 3x1 vector of constants.

        Args:
            drag_coefficients (list[float]): The constant linear drag coefficients used to compute the total drag forces
            affecting the rigid body. The linear drag is given by diag(dx, dy, dz) * [v_x, v_y, v_z] where the velocities
            are expressed in the body frame of the rigid body (using the FRU frame convention).
        """

        # The linear drag coefficients of the vehicle's body frame as a diagonal matrix
        self._drag_coefficients = torch.diag(torch.tensor(drag_coefficients, dtype=torch.float32))

        # The drag force to apply on the vehicle's body frame
        self._drag_force = torch.zeros(3, dtype=torch.float32)

    @property
    def drag(self):
        """The drag force to be applied on the body frame of the vehicle.

        Returns:
            torch.tensor: A tensor with shape (3,) containing the drag force to be applied on the rigid body 
                          according to a FLU body reference frame, expressed in Newton (N) [dx, dy, dz].
        """
        return self._drag_force

    def update(self, state: State, dt: float):
        """Method that updates the drag force to be applied on the body frame of the vehicle. The total drag force
        applied on the body reference frame (FLU convention) is given by diag(dx,dy,dz) * R' * v,
        where v is the velocity of the vehicle expressed in the inertial frame and R' * v = velocity_body_frame.

        Args:
            state: The current state of the vehicle, which contains `linear_body_velocity` as a torch tensor.
            dt (float): The time elapsed between the previous and current function calls (s).

        Returns:
            torch.tensor: A tensor with shape (3,) containing the drag force to be applied on the rigid body 
                          according to a FLU body reference frame.
        """

        # Get the velocity of the vehicle expressed in the body frame of reference
        body_vel = state.linear_velocity_body

        # Compute the component of the drag force to be applied in the body frame
        self._drag_force = -torch.matmul(self._drag_coefficients, body_vel)
        return self._drag_force.to(dtype=torch.float32)

if __name__ == "__main__":
    drag =  LinearDrag([0.50, 0.30, 0.0])
    state = State()
    state.linear_velocity_body = torch.tensor([5.0, -3.0, 2.0])
    dt = 0.01

    drag_force = drag.update(state, dt)
    print("Drag force:", drag_force)