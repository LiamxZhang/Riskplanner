#
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from envs.isaacgym_env import QuadrotorIsaacSim
QIS = QuadrotorIsaacSim()

# setup robot
from configs.configs import ROBOT_PARAMS
from nonlinear_controller import NonlinearController

controller = NonlinearController(
                stage_prefix=ROBOT_PARAMS['stage_prefix'],
                Ki=[0.5, 0.5, 0.5],
                Kr=[2.0, 2.0, 2.0]
            )


from robots.quadrotor import Quadrotor
quadrotor = Quadrotor(**ROBOT_PARAMS, sensors=[], graphical_sensors=[], backends=[controller])
target = [[0.5, 0, 0.2, 0]]

QIS.start()

while QIS.is_running():
    print("current simulation time: ", QIS.time)
    quadrotor.update_trajectory(target)
    QIS.update()
QIS.stop()