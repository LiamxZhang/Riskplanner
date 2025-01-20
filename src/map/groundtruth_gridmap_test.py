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

# Get a local map
# print("real map bounds: ", gt_gridmap.realmap_bounds)
# print("grid map size: ", gt_gridmap.gridmap_size)
import torch 
center = torch.tensor([0,0,0], dtype=torch.float32)
size = torch.tensor([0.5,0.5,0.5], dtype=torch.float32)
localmap = gt_gridmap.get_local_map(center, size)
print("localmap: ", localmap)


while QIS.is_running():
    # print("current simulation time: ", QIS.time)
    gt_gridmap.pub()
    QIS.update()
QIS.stop()