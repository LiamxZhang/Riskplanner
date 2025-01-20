# This script includes the utility functions for the Riskplanner project
#
#
import torch
import numpy as np

def to_world(vertex, transform_matrix):
    """
    Convert vertex in local coordinates to world coordinates
    vertex: list, a vertex of the plane include the coordinate (x, y, z)
    transform_matrix: torch tensor 4*4 
    """
    # Transform vertex into homogeneous coordinate form
    vertex_homogeneous = torch.cat((vertex, torch.tensor([1.0]))) 

    # Perform matrix multiplication and return the first 3 elements
    return torch.matmul(transform_matrix, vertex_homogeneous)[:3]

def is_masked(prim_path, masked_prims):
        # masked_prims = ['Looks','Meshes','Lighting','Road','Buildings','Pavement',
        #                 'GroundPlane','TrafficLight','Bench','Tree','TableChair',
        #                 'Billboard','Lamp','RoadBarriers','Booth','Umbrella','Camera']
        
        for masked in masked_prims:
            if masked in prim_path:
                return True
        return False


def is_point_in_polygon(point, poly_world_vertex):
    """
    Check whether the 3D point is within the polygon 
    point: 1*3 list, include the coordinate (x, y, z)
    poly_world_vertex (list): vertices of polygen in world coordinate
    """
    # Number of vertices
    num_vertices = len(poly_world_vertex)

    angle_sum = 0.0

    for i in range(num_vertices):
        v1 = poly_world_vertex[i] - point
        v2 = poly_world_vertex[(i + 1) % num_vertices] - point

        v1_norm = torch.norm(v1)
        v2_norm = torch.norm(v2)
        dot_product = torch.dot(v1, v2)
        angle = torch.acos(dot_product / (v1_norm * v2_norm))
        angle_sum += angle

    return torch.isclose(angle_sum, torch.tensor(2 * torch.pi, dtype=torch.float32) )

def point_to_plane_distance(point, poly_world_vertex):
    """
    Calculate the distance from point to plane
    point: torch tensor, include the coordinate (x, y, z)
    poly_world_vertex: contain torch tensor, all vertices of the plane
    """
    p0, p1, p2 = poly_world_vertex[0], poly_world_vertex[1], poly_world_vertex[2]

    # Get two vectors
    vec1 = p1 - p0
    vec2 = p2 - p0

    # Calculate the plane normal vector (cross product of vec1 and vec2)
    normal = torch.linalg.cross(vec1, vec2)
    normal = normal / torch.norm(normal)
    
    # Calculate the D in plane Ax + By + Cz + D = 0
    D = -torch.dot(normal, p0)
    
    # Calculate the distance from point to plane
    distance = torch.abs(torch.dot(normal, point) + D) / torch.norm(normal)

    # Get the normal projection
    point_proj = point - (torch.dot(normal, point) + D) * normal
    
    is_inside = is_point_in_polygon(point_proj, poly_world_vertex)

    return distance, is_inside

import trimesh
def point_to_polygon_distance(point, poly_world_vertex):
    """
    Calculate the distance from point to polygon in 3D space
    point: list, include the coordinate (x, y, z)
    poly_world_vertex: contain point list, all vertices of the plane
    """
    # Create the mesh
    n = len(poly_world_vertex)
    if n < 3:
        return float('inf')
    
    faces = []
    # Starting from the 0th vertex, connect the other vertices in sequence to form a triangle
    for i in range(1, n-1):
        face = [0, i, i+1]
        faces.append(face)
    mesh = trimesh.Trimesh(vertices=poly_world_vertex, faces=faces)

    # calculate the distance from point to mesh
    distance = mesh.nearest.signed_distance(point)
    return distance

import matplotlib.pyplot as plt
def plot_3d_points(points_plot):
    """
    Plot a list of 3D points in space.

    Args:
    points_plot (list): A list of lists, where each sublist contains 3D points [x, y, z].
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Separate the x, y, and z coordinates
    x = [point[0] for point in points_plot]
    y = [point[1] for point in points_plot]
    z = [point[2] for point in points_plot]
    
    # Plot the points
    ax.scatter(x, y, z, label="Points Group", alpha=0.6)

    # Add labels and legends
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True)

    plt.show()


def print_prim_and_grid(prim_grid):
    from collections import defaultdict
    grouped_grid = defaultdict(list)

    # put grid_center of same prim into list
    for grid_center, prim in prim_grid.items():
        grouped_grid[prim].append(grid_center)

    for prim, grid_centers in grouped_grid.items():
        # prim_path = str(prim.GetPath())
        # if "Floor" in prim_path:
        print(f"Prim: {prim}")
        print("Grid centers: ", grid_centers)


def traj_tensor(index, traj_list):
    """
    Method that converts the current trajectory point to torch tensor.

    Args:
        index: list index
        traj_list: The list that contains target trajectory points of the vehicle. 
        An example point is given:
            traj = {
                        "position": [x, y, z],       # the target positions [m]
                        "velocity": [vx, vy, vz],    # velocity [m/s]
                        "acceleration": [ax, ay, az],    # accelerations [m/s^2]
                        "jerk": [jx, jy, jz],            # jerk [m/s^3]
                        "yaw_angle": yaw,                # yaw-angle [rad]
                        "yaw_rate": yaw_rate             # yaw-rate [rad/s]
                    }
    """
    traj = traj_list[index]
    p_ref = torch.tensor(traj["position"], dtype=torch.float32)          # 3-element tensor for position
    v_ref = torch.tensor(traj["velocity"], dtype=torch.float32)          # 3-element tensor for velocity
    a_ref = torch.tensor(traj["acceleration"], dtype=torch.float32)      # 3-element tensor for acceleration
    j_ref = torch.tensor(traj["jerk"], dtype=torch.float32)              # 3-element tensor for jerk
    yaw_ref = torch.tensor(traj["yaw_angle"], dtype=torch.float32)       # scalar tensor for yaw
    yaw_rate_ref = torch.tensor(traj["yaw_rate"], dtype=torch.float32)   # scalar tensor for yaw rate
    time = torch.tensor(traj["time"], dtype=torch.float32) 
    return p_ref, v_ref, a_ref, j_ref, yaw_ref, yaw_rate_ref, time


def spherical_to_cartesian(action, r):
    """
    Convert the action space (psi, theta) into spherical coordinates 
    and then to Cartesian coordinates (x, y, z).
    Parameters:
        action (numpy.ndarray): Two dimensions of the action space, psi and theta.
                                action[0]: ψ, in the range [-1, 1]
                                action[1]: θ, in the range [-1, 1]
        r (float): Distance from the point to the origin.
    Returns:
        tuple: Cartesian coordinates (x, y, z).
    """
    # Extract the two angles from the action space
    psi = action[0] * np.pi  # Map to [-π, π]
    theta = (action[1] + 1) * np.pi / 2  # Map to [0, π]

    # Compute Cartesian coordinates
    x = r * np.cos(psi) * np.sin(theta)
    y = r * np.sin(psi) * np.sin(theta)
    z = r * np.cos(theta)

    return x, y, z