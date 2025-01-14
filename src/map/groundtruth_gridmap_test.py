import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from envs.isaacgym_env import QuadrotorIsaacSim

# Make sure that in config.py: CONTROL_PARAMS["grid_resolution"] = 0.2
QIS = QuadrotorIsaacSim()

# Main body
from groundtruth_gridmap import GroundTruthGridMap
gt_gridmap = GroundTruthGridMap()
gt_gridmap.visualize_scatter()

while QIS.is_running():
    # print("current simulation time: ", QIS.time)
    gt_gridmap.pub()
    QIS.update()
QIS.stop()