#import openai
#from arguments import get_config
#from interfaces import setup_LMP
#from visualizers import ValueMapVisualizer
#from envs.rlbench_env import VoxPoserRLBench
#from utils import set_lmp_objects
#import numpy as np
#from rlbench import tasks

#config = get_config('rlbench')
# uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
# for lmp_name, cfg in config['lmp_config']['lmps'].items():
#     cfg['model'] = 'gpt-3.5-turbo'

# initialize env and voxposer ui
#visualizer = ValueMapVisualizer(config['visualizer'])
#env = VoxPoserRLBench(visualizer=visualizer)
#lmps, lmp_env = setup_LMP(env, config, debug=False)
#voxposer_ui = lmps['plan_ui']

# # uncomment this to show all available tasks in rlbench
# # NOTE: in order to run a new task, you need to add the list of objects (and their corresponding env names) to the \"task_object_names.json\" file. See README for more details.
# print([task for task in dir(tasks) if task[0].isupper() and not '_' in task])

# below are the tasks that have object names added to the \"task_object_names.json\" file
# uncomment one to use
#env.load_task(tasks.PutRubbishInBin)
# env.load_task(tasks.LampOff)
# env.load_task(tasks.OpenWineBottle)
# env.load_task(tasks.PushButton)
# env.load_task(tasks.TakeOffWeighingScales)
# env.load_task(tasks.MeatOffGrill)
# env.load_task(tasks.SlideBlockToTarget)
# env.load_task(tasks.TakeLidOffSaucepan)
# env.load_task(tasks.TakeUmbrellaOutOfUmbrellaStand)

#descriptions, obs = env.reset()
#set_lmp_objects(lmps, env.get_object_names())  # set the object names to be used by voxposer

#instruction = np.random.choice(descriptions)
#voxposer_ui(instruction)

from envs.isaacgym_env import QuadrotorIsaacGym

# init environment
CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}
QIG = QuadrotorIsaacGym(CONFIG, task_name=None)

# import the robot
import time
import torch
from robots.quadrotor import Quadrotor
import omni.isaac.core.utils.numpy.rotations as rot_utils

assets_root_path = "/World/envs/env_0"
_crazyflie_position = torch.tensor([-3e4, -1.5e4, 1000.0]) 
_crazyflie_orientation = torch.tensor([0, 0, 0])
copter = Quadrotor(
            prim_path=assets_root_path + "/Crazyflie", name="crazyflie", 
            translation=_crazyflie_position, 
            orientation=rot_utils.euler_angles_to_quats(_crazyflie_orientation, degrees=True)
        )
# 

time.sleep(30)
QIG.stop_task()

