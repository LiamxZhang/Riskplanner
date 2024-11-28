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

controller = NonlinearController(
                stage_prefix=ROBOT_PARAMS['stage_prefix'],
                Ki=[0.5, 0.5, 0.5],
                Kr=[2.0, 2.0, 2.0]
            )

from sensors.lidar import RotatingLidar

lidar = RotatingLidar()

from robots.quadrotor import Quadrotor
quadrotor = Quadrotor(**ROBOT_PARAMS, sensors=[], graphical_sensors=[lidar], backends=[controller])
target = [[0.5, 0, 0.2, 0]]

QIS.start(quadrotor)  # get grid map QIS.prim_grid

# every step get lidar._current_frame


while QIS.is_running():
    print("current simulation time: ", QIS.time)
    quadrotor.update_trajectory(target)
    QIS.update() # App.update()
QIS.stop()