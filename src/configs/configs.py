# Define the default settings for the simulation APP
APP_SETTINGS = {
    "width": 1280,
    "height": 720,
    "sync_loads": True,
    "headless": False,
    "renderer": "RayTracedLighting",
}

# Define the path of the map USD files
MAP_ASSET = {
    "default_usd_path": "/Isaac/Environments/Simple_Room/simple_room.usd",
    "usd_path": "omniverse://localhost/Library/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Environments/NYU/NYC_street.usd",
}

# Define the default settings for the simulation environment
# Default "stage_units_in_meters" is 1.0 m
WORLD_SETTINGS = {
    "physics_dt": 1.0 / 250.0,
    "stage_units_in_meters": 1.0,
    "rendering_dt": 1.0 / 60.0,
    "backend": "torch",
}

# Define the parameters for the crazyflie quadrotor
ROBOT_PARAMS = {
    "stage_prefix": "/World/envs/Crazyflie_00",
    "name": "Crazyflie",
    "usd_path": "/Isaac/Robots/Crazyflie/cf2x.usd", # Under the folder of get_assets_root_path()
    "init_position": [0.0, 0.0, 10.0],
    "init_orientation": [0.0, 0.0, 90.0],
    "scale": [5,5,5],
}



# Define the parameters for the robot controller and planner
CONTROL_PARAMS = {
    "grid_resolution": 1.0,
    "control_cycle": 0.06,
}

