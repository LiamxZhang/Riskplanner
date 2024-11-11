#
#
#
from configs.arguments import get_config
CONFIG = get_config(config_path='./configs/isaacsim_config.yaml')
print(CONFIG)

from envs.isaacgym_env import *

# config = CONFIG['env'], task_name = CONFIG['usd_path']
QIS = QuadrotorIsaacSim(CONFIG, usd_path=CONFIG['usd_path'])

# set ground and if the default world is ROOT
from omni.isaac.core.world import World
world=World()
# world.scene.add_default_ground_plane(prim_path="/World/groundPlane", z_position=CONFIG['env']['groundplane_z'])

# add quadrotor
from tasks.quadrotor_task import QuadrotorTask
quadrotor = QuadrotorTask(config=CONFIG)
quadrotor.set_up_scene(world.scene)
actions = [0.0, 0.5, 1.0]

world.reset()

# program running
timeline = omni.timeline.get_timeline_interface()
timeline.play()
quadrotor.start()
QIS.save_all_prims_in_grid()
QIS.init_timer(CONFIG['control_cycle'])

while QIS.is_running():
    
    # quadrotor.get_camera_image()
    # quadrotor.get_pointcloud()

    # quadrotor.lidar.get_lidar_data(time_step)
    # quadrotor.lidar_local2world()

    # quadrotor.object_detection(QIS.prim_grid, QIS.grid_resolution)

    # prim_name = QIS.get_prim_from_position(quadrotor.get_lidar_data())
    # print("prim_name: ", prim_name)

    quadrotor.get_observations()
    pid_cmd = quadrotor.controller([0.0,0.0,10.0], [0.0,0.0,90.0])
    quadrotor.velocity_action(pid_cmd)
    
    # Control part
    time_step = QIS.get_timestep()
    if time_step:
        print("current simulation time: ", QIS.get_simulation_time())
        quadrotor.control_update(time_step, [[0.25,0.0,10.0,90.0]])
        # quadrotor.nlcontroller.force_and_torques_to_thrust(pid_cmd[0],pid_cmd[1:4])
        breakpoint()
    # quadrotor.camera.get_fov()
    # quadrotor.camera.get_camera_image()
    
    QIS.update()
    
timeline.stop()
QIS.stop_task()
