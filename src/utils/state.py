# 
import torch
from scipy.spatial.transform import Rotation

# Quaternion for rotation between ENU and NED INERTIAL frames
# This rotation is symmetric, so q_ENU_to_NED == q_NED_to_ENU.
# Note: this quaternion follows the convention [qx, qy, qz, qw]
q_ENU_to_NED = torch.tensor([0.70711, 0.70711, 0.0, 0.0])

# A scipy rotation from the ENU inertial frame to the NED inertial frame of reference
rot_ENU_to_NED = Rotation.from_quat(q_ENU_to_NED.numpy())

# Quaternion for rotation between body FLU and body FRD frames
# This rotation is symmetric, so q_FLU_to_FRD == q_FRD_to_FLU.
# Note: this quaternion follows the convention [qx, qy, qz, qw]
q_FLU_to_FRD = torch.tensor([1.0, 0.0, 0.0, 0.0])

# A scipy rotation from the FLU body frame to the FRD body frame
rot_FLU_to_FRD = Rotation.from_quat(q_FLU_to_FRD.numpy())

class State:
    """
    Stores the state of a given vehicle.
    """

    def __init__(self, euler_order: str = "ZYX"):
        """
        Initialize the State object. 
        Inertial frame: ENU: Z-axis up; NED: Z-axis down
            Body frame: FLU: Z-axis up; FRD: Z-axis down
        Default in ENU and FLU
        """

        # The position [x,y,z] of the vehicle's body frame relative to the inertial frame, expressed in the inertial frame
        self.position = torch.tensor([0.0, 0.0, 0.0])

        # The attitude (orientation) of the vehicle's body frame relative to the inertial frame
        # The orientation of the vehicle in quaternion [qx, qy, qz, qw]. Defaults to [1.0, 0.0, 0.0, 0.0].
        self.attitude_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.orient = torch.tensor([0.0, 0.0, 0.0])
        
        # The linear velocity [u,v,w] of the vehicle's body frame expressed in the body frame
        self.linear_velocity_body = torch.tensor([0.0, 0.0, 0.0])

        # The linear velocity [x_dot, y_dot, z_dot] of the vehicle's body frame expressed in the inertial frame
        self.linear_velocity = torch.tensor([0.0, 0.0, 0.0])

        # The angular velocity [wx, wy, wz] of the vehicle's body frame relative to the inertial frame
        self.angular_velocity = torch.tensor([0.0, 0.0, 0.0])

        # The linear acceleration [ax, ay, az] of the vehicle's body frame relative to the inertial frame
        self.linear_acceleration = torch.tensor([0.0, 0.0, 0.0])
        self.euler_order  = euler_order

    def update_state(self, position, attitude_quat, linear_velocity, angular_velocity, linear_acceleration):
        """
        Args:
            position: torch.tensor 
            attitude_quat: quaternion of torch.tensor 
            linear_velocity: torch.tensor, includes 3 linear velocity
            angular_velocity: torch.tensor, includes 3 angular velocity
            linear_acceleration: torch.tensor
        """
        # # update information
        self.position = position
        
        self.attitude_quat = attitude_quat
        self.R = Rotation.from_quat(attitude_quat.numpy())  # input: [qx, qy, qz, qw]
        self.orient = self.rot_to_euler(self.R)  # output angle order depends on euler_order: z,y,x

        # try:
        #     self.linear_velocity = velocity[:, :3]
        #     self.angular_velocity = velocity[:, 3:]
        # except:
        #     pass
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        
        self.linear_acceleration = linear_acceleration
        
        # # update state in body frame
        self.R_body = self.R.inv()
        self.orient_body = self.rot_to_euler(self.R_body)

        self.linear_velocity_body = torch.from_numpy(self.R_body.apply(self.linear_velocity.numpy()))
        self.angular_velocity_body = torch.from_numpy(self.R_body.apply(self.angular_velocity.numpy()))

        return

    def rot_to_euler(self, R):
        # self.euler_order = 'ZYX' or any other order, e.g., 'XYZ', 'YXZ', etc.
        # Get the euler angles as per euler_order
        orient_temp = R.as_euler(self.euler_order, degrees=False)

        # Map to XYZ order
        order_map = {'X': 0, 'Y': 1, 'Z': 2}
        orient_indices = [order_map[axis] for axis in self.euler_order]
        orient_xyz = [orient_temp[i] for i in orient_indices]

        # Convert to torch tensor with XYZ order
        return torch.tensor(orient_xyz)

    def get_position_ned(self):
        """
        Converts the position to the NED convention used by PX4 and other onboard flight controllers
        Returns:
            torch.Tensor: The [x,y,z] of the vehicle in NED convention.
        """
        return torch.tensor(rot_ENU_to_NED.apply(self.position.numpy()))

    def get_attitude_ned_frd(self):
        """
        Converts the attitude of the vehicle to the NED-FRD convention used by PX4 and other onboard flight controllers
        Returns:
            torch.Tensor: The quaternion [qx, qy, qz, qw] for FRD body frame relative to an NED inertial frame.
        """
        attitude_frd_ned = rot_ENU_to_NED * Rotation.from_quat(self.attitude_quat.numpy()) * rot_FLU_to_FRD
        return torch.tensor(attitude_frd_ned.as_quat())

    def get_linear_body_velocity_ned_frd(self):
        """
        Converts the linear body velocity of the vehicle to the NED-FRD convention
        Returns:
            torch.Tensor: The velocity [u,v,w] in the FRD body frame.
        """
        linear_acc_body_flu = Rotation.from_quat(self.attitude_quat.numpy()).inv().apply(self.linear_acceleration.numpy())
        return torch.tensor(rot_FLU_to_FRD.apply(linear_acc_body_flu))

    def get_linear_velocity_ned(self):
        """
        Converts the linear velocity to the NED convention
        Returns:
            torch.Tensor: The velocity [vx,vy,vz] in NED convention.
        """
        return torch.tensor(rot_ENU_to_NED.apply(self.linear_velocity.numpy()))

    def get_angular_velocity_frd(self):
        """
        Converts the angular velocity to the NED-FRD convention
        Returns:
            torch.Tensor: The angular velocity [p,q,r] in the FRD body frame.
        """
        return torch.tensor(rot_FLU_to_FRD.apply(self.angular_velocity.numpy()))

    def get_linear_acceleration_ned(self):
        """
        Converts the linear acceleration to the NED convention
        Returns:
            torch.Tensor: The acceleration [x_ddot, y_ddot, z_ddot] in NED convention.
        """
        return torch.tensor(rot_ENU_to_NED.apply(self.linear_acceleration.numpy()))


if __name__ == "__main__":
    # Initialize test inputs
    position = torch.tensor([1.0, 2.0, 3.0])
    attitude_quat = torch.tensor([0.70711, 0.0, 0.0, 0.70711])  # Quaternion for a 90-degree rotation around X-axis
    linear_velocity = torch.tensor([1.0, 0.5, -0.5])  # 3 linear velocities
    angular_velocity = torch.tensor([0.1, 0.2, 0.3])  # 3 angular velocities
    linear_acceleration = torch.tensor([0.0, 0.0, 0.0])
    # Initialize State object
    vehicle_state = State(euler_order='ZYX')
    
    # Update state
    vehicle_state.update_state(position, attitude_quat, linear_velocity, angular_velocity, linear_acceleration)
    
    # Print state attributes for verification
    print("Position in inertial frame:", vehicle_state.position)
    print("Attitude quaternion in inertial frame:", vehicle_state.attitude_quat)
    print("Orientation (Euler angles) in inertial frame:", vehicle_state.orient)
    print("Linear velocity in inertial frame:", vehicle_state.linear_velocity)
    print("Angular velocity in inertial frame:", vehicle_state.angular_velocity)
    
    print("Orientation (Euler angles) in body frame:", vehicle_state.orient_body)
    print("Linear velocity in body frame:", vehicle_state.linear_velocity_body)
    print("Angular velocity in body frame:", vehicle_state.angular_velocity_body)