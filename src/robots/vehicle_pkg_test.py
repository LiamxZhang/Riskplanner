import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from envs.isaacgym_env import QuadrotorIsaacSim
from configs.configs import ROBOT_PARAMS
QIS = QuadrotorIsaacSim()

# Crazyflie USD file
# from omni.isaac.core.utils.nucleus import get_assets_root_path
# assets_root_path = get_assets_root_path()
# usd_path = assets_root_path + "/Isaac/Robots/Crazyflie/cf2x.usd"

# Iris USD file
# usd_path = "omniverse://localhost/Library/NVIDIA/Assets/Isaac/4.2/Isaac/Robots/Iris/iris.usd"
usd_path = ROBOT_PARAMS["usd_path"]
stage_prefix = "/World/envs/Iris"

from vehicle import Vehicle
vehicle = Vehicle(stage_prefix=stage_prefix,usd_path = usd_path, scale=[1.0,1.0,1.0])

QIS.reset()

print("Prim: ", vehicle.prim)

while QIS.is_running():
    # print("current simulation time: ", QIS.time)
    
    QIS.update()
QIS.stop()