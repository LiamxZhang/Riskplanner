import sys
sys.path.append("..")

from envs.isaacgym_env import QuadrotorIsaacSim

QIS = QuadrotorIsaacSim()

# 
QIS.start()

# Main body
# from map_manager import MapManager
# Map = MapManager()

from groundtruth_gridmap import GroundTruthGridMap
Map = GroundTruthGridMap()

while QIS.is_running():
    # print("current simulation time: ", QIS.time)
    
    QIS.update()
QIS.stop()