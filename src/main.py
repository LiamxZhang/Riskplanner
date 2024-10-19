#
#
#
# from isaacsim import SimulationApp
from omni.isaac.kit import SimulationApp
# default configuration
CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}
kit = SimulationApp(launch_config=CONFIG)

# load stage
import sys
import omni
import carb
from omni.isaac.core.utils.nucleus import get_assets_root_path, is_file

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    kit.close()
    sys.exit()
# Set task directory
usd_path = assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd"
#usd_path = "omniverse://localhost/Library/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Environments/NYU/NYC_sim.usd"

# Load the stage
if is_file(usd_path):
    omni.usd.get_context().open_stage(usd_path)
else:
    carb.log_error(
        f"the usd path {usd_path} could not be opened, please make sure that {usd_path} is a valid usd file in {assets_root_path}"
    )
    kit.close()
    sys.exit()
# Wait two frames so that stage starts loading
kit.update()
kit.update()
print("Loading stage...")

from omni.isaac.core.utils.stage import is_stage_loading
while is_stage_loading():
    kit.update()
print("Loading Complete")
# Wait for the program start

# set ground and if the default world is ROOT
from omni.isaac.core.world import World
world=World()
world.scene.add_default_ground_plane(prim_path="/World/groundPlane", z_position=-2.0)

# add quadrotor
import torch
from robots.quadrotor import Quadrotor
import omni.isaac.core.utils.numpy.rotations as rot_utils

assets_root_path = "/World/envs/env_0"
_crazyflie_position = torch.tensor([0, 0, 0.0]) 
_crazyflie_orientation = torch.tensor([0, 0, 0])
copter = Quadrotor(
            prim_path=assets_root_path + "/Crazyflie", name="crazyflie", 
            translation=_crazyflie_position, 
            orientation=rot_utils.euler_angles_to_quats(_crazyflie_orientation, degrees=True)
        )
world.reset()
# add multiple quadrotors

# robot move


# add camera
import numpy as np
import matplotlib.pyplot as plt
from omni.isaac.sensor import Camera

camera = Camera(
    prim_path="/World/camera",
    position=np.array([0.0, 0.0, 0.0]),
    frequency=20,
    resolution=(256, 256),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
)
camera.initialize()
camera.add_motion_vectors_to_frame()
i = 0

# attach camera to robot


# Run until closed
omni.timeline.get_timeline_interface().play()
while kit._app.is_running() and not kit.is_exiting():
    world.step(render=True)
    # input = carb.input.acquire_input_interface()
    # Run in realtime mode, we don't specify the step size

    print(camera.get_current_frame())
    if i == 100:
        points_2d = camera.get_image_coords_from_world_points(
            np.array([copter.get_world_pose()[0], copter.get_world_pose()[0]])
        )
        points_3d = camera.get_world_points_from_image_coords(points_2d, np.array([24.94, 24.9]))
        print("points_2d", points_2d)
        print("points_3d: ", points_3d)
        imgplot = plt.imshow(camera.get_rgba()[:, :, :3])
        plt.show()

        # point_cloud = camera.get_pointcloud()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='r', marker='o')
        # plt.show()

        print(camera.get_current_frame()["motion_vectors"])
    # if world.is_playing():
    #     if world.current_time_step_index == 0:
    #         world.reset()
    i += 1
    kit.update()
omni.timeline.get_timeline_interface().stop()
kit.close()