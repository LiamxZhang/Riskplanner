import sys
sys.path.append("..")

from envs.isaacgym_env import QuadrotorIsaacSim

QIS = QuadrotorIsaacSim()

# Crazyflie USD file
from omni.isaac.core.utils.nucleus import get_assets_root_path
assets_root_path = get_assets_root_path()
usd_path = assets_root_path + "/Isaac/Robots/Crazyflie/cf2x.usd"

from vehicle import Vehicle

vehicle = Vehicle(stage_prefix="/World/envs/env_00",usd_path = usd_path, scale=[5.0,5.0,5.0])

QIS.start()

while QIS.is_running():
    # print("current simulation time: ", QIS.time)
    
    QIS.update()
QIS.stop()