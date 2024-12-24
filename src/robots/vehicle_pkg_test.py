import sys
sys.path.append("..")

from envs.isaacgym_env import QuadrotorIsaacSim
from configs.configs import ROBOT_PARAMS
QIS = QuadrotorIsaacSim()

# Crazyflie USD file
from omni.isaac.core.utils.nucleus import get_assets_root_path
assets_root_path = get_assets_root_path()
# usd_path = assets_root_path + "/Isaac/Robots/Crazyflie/cf2x.usd"

# Iris USD file
# usd_path = "omniverse://localhost/Library/NVIDIA/Assets/Isaac/4.2/Isaac/Robots/Iris/iris.usd"
usd_path = ROBOT_PARAMS["usd_path"]

# # Local USD file path
# import os
# from pathlib import Path
# EXTENSION_FOLDER_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
# ROOT = str(EXTENSION_FOLDER_PATH.parent.resolve())
# print("ROOT: ", ROOT)
# usd_path = ROOT + "/assets/robots/Iris/iris.usd"

from vehicle import Vehicle

vehicle = Vehicle(stage_prefix=ROBOT_PARAMS["stage_prefix"],usd_path = usd_path, scale=[1.0,1.0,1.0])

QIS.start()

while QIS.is_running():
    # print("current simulation time: ", QIS.time)
    
    QIS.update()
QIS.stop()