import sys
sys.path.append("..")

from envs.isaacgym_env import QuadrotorIsaacSim

QIS = QuadrotorIsaacSim()

# 
QIS.start()

# Main body
from groundtruth_gridmap import GroundTruthGridMap
gt_gridmap = GroundTruthGridMap()

while QIS.is_running():
    # print("current simulation time: ", QIS.time)
    
    QIS.update()
QIS.stop()