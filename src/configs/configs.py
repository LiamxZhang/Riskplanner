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
    "NYC_usd_path": "omniverse://localhost/Library/NVIDIA/Assets/Isaac/4.2/Isaac/Environments/NYU/NYC_street.usd",
    "masked_prims": {
                        "default_usd_path": ['Looks','Light','Floor','Towel_Room01','GroundPlane'], # 'table'
                        "NYC_usd_path": ['Looks','Meshes','Lighting','Road','Pavement',
                                    'GroundPlane','TrafficLight','Bench','Tree','TableChair',
                                    'Billboard','Lamp','RoadBarriers','Booth','Umbrella','Camera'],  # 'Buildings', 'Car'
                    },
    # Parameters
    "ros2_publish": True,
    "max_length_of_pointset": int(1e4),
    "extend_units": int(5),
    "max_fill_value": float(5),
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
    "usd_path": "omniverse://localhost/Library/NVIDIA/Assets/Isaac/4.2/Isaac/Robots/Iris/iris.usd",
    "init_position": [-1.0, 0.0, 1.0],
    "init_orientation": [0.0, 0.0, 0.0, 1.0],
    "scale": [1,1,1],
}

# Define the parameters for the robot controller and planner
CONTROL_PARAMS = {
    "grid_resolution": 0.2,
    "control_cycle": 0.5,   # valid min 0.06, but effective min 0.5
    "num_rotors": 4,
    "mass": 1.50,    # Mass in Kg
    "gravity": 9.81, # The gravity acceleration ms^-2
    "target_position": [1.5, 0.0, 1.0],
    "target_radius": 0.1 
}

# Iris drone risk parameters
RISK_PARAMS = {
    "impact_area": 0.0188, # The impact area m^2 
    "population_density": 8.358e-3, # The population density people/m^2
    "crash_probability": 3.42e-4, # The crash probability /hour
    "drag_coefficient": 0.3, 
    "air_density": 1.225, # The air density
    "casualty": 0.27, # The number of casualties caused by an average traffic accident
    "fatality_risk": 0.5, # The weight factor of fatality risk
    "damage_risk": 0.25, # The weight factor of property damage risk
    "noise_impact": 0.25, # The weight factor of noise impact 
}

# Define the parameters for the LiDAR sensor 
LIDAR_PARAMS = {
    "type": "Lidar",
    "frequency": 20.0,
    "range": [0.4, 700.0],
    "fov": [90.0, 120.0],   # horizontal_fov 90, vertical_fov 120
    "resolution": [10.0, 10.0],
    "rotation_rate": 0.0,
    "orientation": [0.0, 0.0, 0.0],
    "translation": [0.0, 0.0, -0.5],
    "draw_lines": False,
    "display_points": True,
    "multi_line_mode": True,
    "enable_semantics": True,
    "visualization": False,
    "ros2_publish": False,
}
