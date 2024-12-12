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
    "stage_prefix": "/World/envs/Iris_00",
    "name": "Iris",
    # "usd_path": "/Isaac/Robots/Crazyflie/cf2x.usd", # Under the folder of get_assets_root_path()
    "usd_path": "omniverse://localhost/Library/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Robots/Iris/iris.usd",
    "init_position": [0.0, 0.0, 0.2],
    "init_orientation": [0.0, 0.0, 0.0, 1.0],
    "scale": [1,1,1],
}

# Define the parameters for the robot controller and planner
CONTROL_PARAMS = {
    "grid_resolution": 0.2,
    "control_cycle": 0.5,   # valid min 0.06, but effective min 0.5
    "num_rotors": 4,
}

# Define the parameters for the LiDAR sensor 
LIDAR_PARAMS = {
    "type": "Lidar",
    "frequency": 20.0,
    "range": [0.4, 1000.0],
    "fov": [90.0, 1.0],
    "resolution": [0.4, 0.4],
    "rotation_rate": 0.0,
    "orientation": [0.0, 0.0, 0.0],
    "translation": [0.0, 0.0, -0.2],
    "draw_lines": False,
    "display_points": True,
    "multi_line_mode": True,
    "enable_semantics": True,
    "visualization": False,
}
