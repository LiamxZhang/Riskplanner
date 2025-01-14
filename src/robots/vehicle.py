#!/usr/bin/env python
# Numerical computations
import torch

# Low level APIs
import carb
from pxr import Usd, Gf

# High level Isaac sim APIs
import omni.usd
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.usd import get_stage_next_free_path
from omni.isaac.core.robots.robot import Robot
from omni.isaac.dynamic_control import _dynamic_control

# Extension APIs
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from utils.state import State
from envs.isaacgym_env import QuadrotorIsaacSim

def get_world_transform_xform(prim: Usd.Prim):
    """
    Get the local transformation of a prim using omni.usd.get_world_transform_matrix().
    See https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd/omni.usd.get_world_transform_matrix.html
    Args:
        prim (Usd.Prim): The prim to calculate the world transformation.
    Returns:
        A tuple of:
        - Translation vector.
        - Rotation quaternion, i.e. 3d vector plus angle.
        - Scale vector.
    """
    world_transform: Gf.Matrix4d = omni.usd.get_world_transform_matrix(prim)
    rotation: Gf.Rotation = world_transform.ExtractRotation()
    return rotation


class Vehicle(Robot):
    
    def __init__(
        self,
        stage_prefix: str = "/quadrotor",
        usd_path: str = None,
        init_pos=[0.0, 0.0, 0.0],
        init_orient=[0.0, 0.0, 0.0, 1.0],
        scale=[1.0, 1.0, 1.0],
        sensors=[],
        graphical_sensors=[],
        backends=[]
    ):
        """
        Class that initializes a vehicle in the isaac sim's curent stage

        Args:
            stage_prefix (str): The name the vehicle will present in the simulator when spawned. Defaults to "quadrotor".
            usd_path (str): The USD file that describes the looks and shape of the vehicle. Defaults to "".
            init_pos (list): The initial position of the vehicle in the inertial frame (in ENU convention). Defaults to [0.0, 0.0, 0.0].
            init_orient (list): The initial orientation of the vehicle in quaternion [qx, qy, qz, qw]. Defaults to [0.0, 0.0, 0.0, 1.0].
        """

        # Get the current world at which we want to spawn the vehicle
        self._world = QuadrotorIsaacSim().world
        self._current_stage = self._world.stage
        
        # Save the name with which the vehicle will appear in the stage
        # and the name of the .usd file that contains its description
        self._stage_prefix = get_stage_next_free_path(self._current_stage, stage_prefix, False)
        self._usd_file = usd_path
        
        # Get the vehicle name by taking the last part of vehicle stage prefix
        self._vehicle_name = self._stage_prefix.rpartition("/")[-1]

        # Spawn the vehicle primitive in the world's stage
        # self._prim = define_prim(self._stage_prefix, "Xform")
        # self._prim = get_prim_at_path(self._stage_prefix)
        # self._prim.GetReferences().AddReference(self._usd_file)
        add_reference_to_stage(self._usd_file, self._stage_prefix)

        self.init_pos = init_pos
        self.init_orient = [init_orient[3], init_orient[0], init_orient[1], init_orient[2]]
        # Initialize the "Robot" class
        # Note: we need to change the rotation to have qw first, because NVidia
        # does not keep a standard of quaternions inside its own libraries (not good, but okay)
        super().__init__(
            prim_path=self._stage_prefix,
            name=self._vehicle_name,
            position=self.init_pos,
            orientation=self.init_orient, # [w,x,y,z]
            scale=scale,
            articulation_controller=None,
        )

        self._vehicle_dc_interface = None

        # Add this object for the world to track, so that if we clear the world, this object is deleted from memory and
        # as a consequence, from the VehicleManager as well
        self._world.scene.add(self)
        
        # Variable that will hold the current state of the vehicle
        self._state = State(euler_order='ZYX')
        
        # Add a callback to the physics engine to update the current state of the system
        self._world.add_physics_callback(self._stage_prefix + "/state", self.update_state)

        # Add the update method to the physics callback if the world was received
        # so that we can apply forces and torques to the vehicle. Note, this method should        
        # be implemented in classes that inherit the vehicle object
        self._world.add_physics_callback(self._stage_prefix + "/update", self.update)

        # --------------------------------------------------------------------
        # -------------------- Add sensors to the vehicle --------------------
        # --------------------------------------------------------------------
        self._sensors = sensors
        
        for sensor in self._sensors:
            sensor.initialize(self)

        # Add callbacks to the physics engine to update each sensor at every timestep
        # and let the sensor decide depending on its internal update rate whether to generate new data
        self._world.add_physics_callback(self._stage_prefix + "/Sensors", self.update_sensors)

        # --------------------------------------------------------------------
        # -------------------- Add the graphical sensors to the vehicle ------
        # --------------------------------------------------------------------
        self._graphical_sensors = graphical_sensors

        for graphical_sensor in self._graphical_sensors:
            graphical_sensor.initialize(self)

        # Add callbacks to the rendering engine to update each graphical sensor at every timestep of the rendering engine
        self._world.add_render_callback(self._stage_prefix + "/GraphicalSensors", self.update_graphical_sensors)

        # --------------------------------------------------------------------
        # -------------------- Add control backends to the vehicle -----------
        # --------------------------------------------------------------------
        self._backends = backends

        # Initialize the backends
        for backend in self._backends:
            backend.initialize(self)

        # Add a callbacks for the backends
        self._world.add_physics_callback(self._stage_prefix + "/Control_state", self.update_sim_state)


    """
    Properties
    """

    @property
    def state(self):
        """The state of the vehicle.

        Returns:
            State: The current state of the vehicle, i.e., position, orientation, linear and angular velocities...
        """
        return self._state
    
    @property
    def vehicle_name(self) -> str:
        """Vehicle name.

        Returns:
            Vehicle name (str): last prim name in vehicle prim path
        """
        return self._vehicle_name

    @property
    def stage_prefix(self) -> str:
        """Stage prefix.

        Returns:
            Stage prefix path (str): The stage prefix path of current vehicle
        """
        return self._stage_prefix
    
    """
    Operations
    """

    def apply_force(self, force, pos=[0.0, 0.0, 0.0], body_part="/body"):
        """
        Method that will apply a force on the rigidbody, on the part specified in the 'body_part' at its relative position
        given by 'pos' (following a FLU) convention. 

        Args:
            force (list): A 3-dimensional vector of floats with the force [Fx, Fy, Fz] on the body axis of the vehicle according to a FLU convention.
            pos (list): _description_. Defaults to [0.0, 0.0, 0.0].
            body_part (str): . Defaults to "/body".
        """

        # Get the handle of the rigidbody that we will apply the force to
        rb = self.get_dc_interface().get_rigid_body(self._stage_prefix + body_part)

        # Apply the force to the rigidbody. The force should be expressed in the rigidbody frame
        self.get_dc_interface().apply_body_force(rb, carb._carb.Float3(force), carb._carb.Float3(pos), False)

    def apply_torque(self, torque, body_part="/body"):
        """
        Method that when invoked applies a given torque vector to /<rigid_body_name>/"body" or to /<rigid_body_name>/<body_part>.

        Args:
            torque (list): A 3-dimensional vector of floats with the force [Tx, Ty, Tz] on the body axis of the vehicle according to a FLU convention.
            body_part (str): . Defaults to "/body".
        """

        # Get the handle of the rigidbody that we will apply a torque to
        rb = self.get_dc_interface().get_rigid_body(self._stage_prefix + body_part)

        # Apply the torque to the rigidbody. The torque should be expressed in the rigidbody frame
        self.get_dc_interface().apply_body_torque(rb, carb._carb.Float3(torque), False)


    def update_state(self, dt: float):
        """
        Method that is called at every physics step to retrieve and update the current state of the vehicle, i.e., get
        the current position, orientation, linear and angular velocities, and acceleration of the vehicle.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Get the body frame interface of the vehicle (this will be the frame used to get the position, orientation, etc.)
        body = self.get_dc_interface().get_rigid_body(self._stage_prefix + "/body")

        # Get the current position and orientation in the inertial frame
        pose = self.get_dc_interface().get_rigid_body_pose(body)

        # Get the attitude according to the convention [w, x, y, z]
        prim = self._world.stage.GetPrimAtPath(self._stage_prefix + "/body")
        rotation_quat = get_world_transform_xform(prim).GetQuaternion()
        rotation_quat_real = rotation_quat.GetReal()
        rotation_quat_img = rotation_quat.GetImaginary()
        # Get the quaternion according to the [qx, qy, qz, qw] standard
        attitude_quat = torch.tensor(
            [rotation_quat_img[0], rotation_quat_img[1], rotation_quat_img[2], rotation_quat_real], dtype=torch.float32
        )
        
        # The linear velocity [x_dot, y_dot, z_dot] of the vehicle's body frame expressed in the inertial frame of reference
        linear_vel = torch.tensor(self.get_dc_interface().get_rigid_body_linear_velocity(body), dtype=torch.float32)
        # Get the angular velocity of the vehicle expressed in the body frame of reference
        ang_vel = torch.tensor(self.get_dc_interface().get_rigid_body_angular_velocity(body), dtype=torch.float32)

        # Get the linear acceleration of the body relative to the inertial frame, expressed in the inertial frame, X_ddot = [x_ddot, y_ddot, z_ddot]
        # Note: we must do this approximation, since the Isaac sim does not output the acceleration of the rigid body directly
        linear_acceleration = (linear_vel - self._state.linear_velocity) / dt

        # Update the state variables
        self._state.update_state(torch.tensor(pose.p, dtype=torch.float32), attitude_quat, linear_vel, ang_vel, linear_acceleration)

    def start(self):
        """
        Method that should be implemented by the class that inherits the vehicle object.
        """
        pass

    def stop(self):
        """
        Method that should be implemented by the class that inherits the vehicle object.
        """
        pass

    def update(self, dt: float):
        """
        Method that computes and applies the forces to the vehicle in
        simulation based on the motor speed. This method must be implemented
        by a class that inherits this type and it's called periodically by the physics engine.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """
        pass

    def reset(self):
        """
        Method that should be implemented by the class that inherits the vehicle object.
        This method is expected to reset the vehicle's state to its initial conditions.
        
        Example use cases:
        - Resetting position, orientation, and velocity of the vehicle to default values.
        - Reinitializing control parameters or any internal state variables.

        """
        pass

    def update_sensors(self, dt: float):
        """Callback that is called at every physics steps and will call the sensor.update method to generate new
        sensor data. For each data that the sensor generates, the backend.update_sensor method will also be called for
        every backend. For example, if new data is generated for an IMU and we have a PX4MavlinkBackend, then the update_sensor
        method will be called for that backend so that this data can latter be sent thorugh mavlink.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Call the update method for the sensor to update its values internally (if applicable)
        for sensor in self._sensors:
            sensor_data = sensor.update(self._state, dt)


    def update_graphical_sensors(self, event):
        """Callback that is called at every rendering steps and will call the graphical_sensor.update method to generate new
        sensor data. For each data that the sensor generates, the backend.update_graphical_sensor method will also be called for
        every backend. For example, if new data is generated for a monocular camera and we have a ROS2Backend, then the update_graphical_sensor
        method will be called for that backend so that this data can latter be sent through a ROS2 topic.

        Args:
            event (float): The timer event that contains the time elapsed between the previous and current function calls (s).
        """

        # Call the update method for the sensor to update its values internally (if applicable)
        for sensor in self._graphical_sensors:
            sensor_data = sensor.update(self._state, event.payload['dt'])

    def update_sim_state(self, dt: float):
        """
        Callback that is used to "send" the current state for each backend being used to control the vehicle. This callback
        is called on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """
        for backend in self._backends:
            backend.update_state(self._state)

    def get_dc_interface(self):
        """Get the dynamic_control_interface."""
        if self._vehicle_dc_interface is None:
            self._vehicle_dc_interface = _dynamic_control.acquire_dynamic_control_interface()

        return self._vehicle_dc_interface
