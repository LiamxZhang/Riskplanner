import sys
sys.path.append("..")

from configs.configs import ROBOT_PARAMS
from envs.isaacgym_env import QuadrotorIsaacSim

QIS = QuadrotorIsaacSim()

##
from quadrotor import Quadrotor

quadrotor = Quadrotor(**ROBOT_PARAMS, sensors=[], graphical_sensors=[], backends=[])
dt = 0.01

## main
QIS.start()

while QIS.is_running():
    # print("current simulation time: ", QIS.time)
    # quadrotor.update(dt)
    QIS.update()
QIS.stop()