import datetime
import os
import time
import gym
import hydra
import torch
import omni
from omni.isaac.kit import SimulationApp


CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}

# Simple example showing how to start and stop the helper
kit = SimulationApp(launch_config=CONFIG)

### Perform any omniverse imports here after the helper loads ###
# Locate Isaac Sim assets folder to load sample
import carb
from omni.isaac.core.utils.nucleus import get_assets_root_path, is_file

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    kit.close()
    sys.exit()
usd_path = assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd"

if is_file(usd_path):
    omni.usd.get_context().open_stage(usd_path)
else:
    carb.log_error(
        f"the usd path {usd_path} could not be opened, please make sure that {args.usd_path} is a valid usd file in {assets_root_path}"
    )
    kit.close()
    sys.exit()
# Wait two frames so that stage starts loading
kit.update()
kit.update()

print("Loading stage...")
from omni.isaac.core.utils.stage import is_stage_loading

while is_stage_loading():
    kit.update()
print("Loading Complete")

# after the map is loaded
from omni.isaac.occupancy_map import _occupancy_map
import omni.isaac.core.utils.prims as prims_utils
from riskmap import RiskMap
rmap = RiskMap()


omni.timeline.get_timeline_interface().play()
# Run in test mode, exit after a fixed number of steps
if True:
    for i in range(1000):
        # rmap = RiskMap()
        rmap.get_point_cloud()
        # rmap.pub()
        # rmap.get_prim_pos()
        # rmap.build_point_cloud()
        rmap.pub_risk_map()

        # physx = omni.physx.acquire_physx_interface()
        # stage_id = omni.usd.get_context().get_stage_id()

        # generator = _occupancy_map.Generator(physx, stage_id)
        # # 0.05m cell size, output buffer will have 4 for occupied cells, 5 for unoccupied, and 6 for cells that cannot be seen
        # # this assumes your usd stage units are in m, and not cm
        # generator.update_settings(.05, 4, 5, 6)
        # # Set location to map from and the min and max bounds to map to
        # generator.set_transform((0, 0, 0), (-2, -2, 0), (2, 2, 0))
        # generator.generate2d()
        # # Get locations of the occupied cells in the stage
        # points = generator.get_occupied_positions()
        # # Get computed 2d occupancy buffer
        # buffer = generator.get_buffer()
        # # Get dimensions for 2d buffer
        # dims = generator.get_dimensions()

        # print("Points: ", points)
        # predicate = lambda path: prims_utils.get_prim_type_name(path) == "Floor"
        # child_prim = prims_utils.get_first_matching_child_prim("/", predicate)
        predicate = lambda path: True
        child_prim = prims_utils.get_all_matching_child_prims("/", predicate)
        # print("child_prim: ", child_prim)
        attribute_names = prims_utils.get_prim_attribute_names("/Root/Floor/SM_Template_Map_Floor")
        # print("attribute_names: ", attribute_names)
        prim = prims_utils.get_prim_at_path("/Root/Floor/SM_Template_Map_Floor")
        # print("prim: ", prim)
        # print("prim attribute: ", prim.GetAttribute("xformOp:translate").Get())

        translate = prims_utils.get_prim_attribute_value("/Root/Floor", attribute_name="xformOp:translate")
        # print("translate: ", translate)
        scale = prims_utils.get_prim_attribute_value("/Root/Floor", attribute_name="xformOp:scale")
        # print("scale: ", scale)
        
        time.sleep(1)

        # Run in realtime mode, we don't specify the step size
        kit.update()
    # rmap.pub()
    rmap.pub_end()
else:
    while kit.is_running():
        # Run in realtime mode, we don't specify the step size
        kit.update()

omni.timeline.get_timeline_interface().stop()


kit.close()  # Cleanup application

