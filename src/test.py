#
#
#
from configs.arguments import get_config
CONFIG = get_config(config_path='./configs/isaacsim_config.yaml')
print(CONFIG)

import time
from envs.isaacgym_env import *

# config = CONFIG['env'], task_name = CONFIG['usd_path']
QIS = QuadrotorIsaacSim(CONFIG['app'], usd_path=CONFIG['usd_path'])

# set ground and if the default world is ROOT
from omni.isaac.core.world import World
world=World()
world.scene.add_default_ground_plane(prim_path="/World/groundPlane", z_position=CONFIG['env']['groundplane_z'])

# add quadrotor
from tasks.quadrotor_task import QuadrotorTask
quadrotor = QuadrotorTask(config=CONFIG)
quadrotor.set_up_scene(world.scene)
quadrotor.get_observations()
actions = [0.0, 0.5, 1.0]

world.reset()

# program running
timeline = omni.timeline.get_timeline_interface()
timeline.play()
QIS.save_all_prims_in_grid()

while QIS.is_running():
    time_step = timeline.get_end_time()
    # quadrotor.get_camera_image()
    # quadrotor.get_pointcloud()
    quadrotor.lidar.get_lidar_data(time_step)
    quadrotor.lidar_local2world()
    # quadrotor.object_detection(QIS.prim_grid, 0.5)

    prim_name = QIS.get_prim_from_position(quadrotor.get_lidar_data())
    # print("prim_name: ", prim_name)

    quadrotor.get_observations()
    pid_cmd = quadrotor.controller([1.5,0.0,60.0], [0.0,0.0,0.0])
    quadrotor.velocity_action(pid_cmd)
    # quadrotor.camera.get_fov()
    # quadrotor.camera.get_camera_image()
    QIS.update()
    # time.sleep(0.1)
timeline.stop()
QIS.stop_task()
