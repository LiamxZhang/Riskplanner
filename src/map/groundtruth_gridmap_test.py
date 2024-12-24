import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

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