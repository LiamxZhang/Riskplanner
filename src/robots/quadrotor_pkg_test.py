import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from configs.configs import ROBOT_PARAMS
from envs.isaacgym_env import QuadrotorIsaacSim

QIS = QuadrotorIsaacSim()

##
from quadrotor import Quadrotor

quadrotor_params ={
    "stage_prefix": "/World/envs/Iris",
    "name": "Iris",
    "usd_path": "omniverse://localhost/Library/NVIDIA/Assets/Isaac/4.2/Isaac/Robots/Iris/iris.usd",
    "init_position": [0.0, 0.0, 0.2],
    "init_orientation": [0.0, 0.0, 0.0, 1.0],
    "scale": [1,1,1],
}

quadrotor = Quadrotor(**quadrotor_params, sensors=[], graphical_sensors=[], backends=[])

## main
QIS.reset()

while QIS.is_running():
    # print("current simulation time: ", QIS.time)
    # quadrotor.update(dt)
    QIS.update()
QIS.stop()