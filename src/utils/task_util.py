# This script includes the utility functions for the Riskplanner project
#
#
import numpy as np

def to_world(vertex, transform_matrix):
    """
    Convert vertex in local coordinates to world coordinates
    vertex: list, a vertex of the plane include the coordinate (x, y, z)
    transform_matrix: numpy array 4*4 
    """
    import numpy as np

    vertex_homogeneous = np.append(vertex, 1)  # 将顶点变为齐次坐标形式
    return np.dot(transform_matrix, vertex_homogeneous)[:3]

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
    poly_world_vertex: vertices of polygen in world coordinate
    """
    # number of vertices
    num_vertices = len(poly_world_vertex)

    angle_sum = 0.0

    for i in range(num_vertices):
        v1 = poly_world_vertex[i] - point
        v2 = poly_world_vertex[(i + 1) % num_vertices] - point

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        dot_product = np.dot(v1, v2)
        angle = np.arccos(dot_product / (v1_norm * v2_norm))
        angle_sum += angle

    return np.isclose(angle_sum, 2 * np.pi)

def point_to_plane_distance(point, poly_world_vertex):
    """
    calculate the distance from point to plane
    point: numpy array, include the coordinate (x, y, z)
    poly_world_vertex: contain numpy array, all vertices of the plane
    """
    p0, p1, p2 = poly_world_vertex[0], poly_world_vertex[1], poly_world_vertex[2]

    # get two vectors
    vec1 = p1 - p0
    vec2 = p2 - p0
    # Calculate the plane normal vector (cross product of vec1 and vec2)
    normal = np.cross(vec1, vec2)
    normal = normal / np.linalg.norm(normal)
    
    # calculate the D in plane Ax + By + Cz + D = 0
    D = -np.dot(normal, p0)
    
    # calculate the distance from point to plane
    distance = np.abs(np.dot(normal, point) + D) / np.linalg.norm(normal)

    # get the normal projection
    point_proj = point - (np.dot(normal, point) + D) * normal
    
    is_inside = is_point_in_polygon(point_proj, poly_world_vertex)

    return distance, is_inside

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