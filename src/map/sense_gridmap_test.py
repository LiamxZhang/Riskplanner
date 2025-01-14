# 
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from envs.isaacgym_env import QuadrotorIsaacSim
QIS = QuadrotorIsaacSim()

# setup robot
from configs.configs import ROBOT_PARAMS
from controller.nonlinear_controller import NonlinearController

quadrotor_params ={
    "stage_prefix": "/World/envs/Iris",
    "name": "Iris",
    "usd_path": "omniverse://localhost/Library/NVIDIA/Assets/Isaac/4.2/Isaac/Robots/Iris/iris.usd",
    "init_position": [-1.5, 0.0, 0.2],
    "init_orientation": [0.0, 0.0, 0.0, 1.0],
    "scale": [1,1,1],
}

controller = NonlinearController(
                stage_prefix=quadrotor_params['stage_prefix'],
                Ki=[0.5, 0.5, 0.5],
                Kr=[2.0, 2.0, 2.0]
            )

from sensors.lidar import RotatingLidar
lidar = RotatingLidar()

from map.sense_gridmap import SenseGridMap
sense_gridmap = SenseGridMap()
# quadrotor.add_backends(sense_gridmap)

from robots.quadrotor import Quadrotor
quadrotor = Quadrotor(**quadrotor_params, sensors=[], graphical_sensors=[lidar], backends=[controller, sense_gridmap])

QIS.reset() 

# Set the vehicle init position as [x, y, z, psi]
target = [[-1.0, 0, 0.2, 0.0]]

count = 0

while QIS.is_running():
    count += 1
    # print("current simulation time: ", QIS.time)
    quadrotor.update_trajectory(target)
    QIS.update() # App.update()
    if count==100:
        quadrotor.reset()
        count = 0
    
QIS.stop()