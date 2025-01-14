from omni.physx import get_physx_cooking_interface
from pxr import UsdGeom, Usd, Gf
import torch
import trimesh
from scipy.spatial import Delaunay
import omni.usd 
from omni.physx import PhysicsSchemaTools 
from omni.physx import PhysxCollisionRepresentationResult 

class CollisionChecker:
    def __init__(self, stage, length_unit):
        self._stage = stage
        self.length_unit = length_unit
        self.threshold = None
        self.transform_matrix = None
        self.prim_path = None
        # self.convex_hulls = None
        # self.result = None

    def is_collided_with_prim(self, prim, point, threshold=None):
        """
        Judge whether the point collides with the prim
        Args: 
            threshold: it is a collision, only when the point-prim distance < the threshold
        Returns the flag: whether point collides with the prim
        """
        self.point = point
        # Transform the prim to world space
        self.transform_matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        self.prim_path = str(prim.GetPath())
        
        # Set default threshold if not provided
        if threshold is None:
            self.threshold = 1e-4 / self.length_unit
        else:
            self.threshold = threshold

        # Request convex collision representation
        physx_cooking = get_physx_cooking_interface()        
        stage_id = omni.usd.get_context().get_stage_id()
        prim_id = PhysicsSchemaTools.sdfPathToInt(self.prim_path)  # Get prim ID

        # Asynchronous request for convex collision representation
        physx_cooking.request_convex_collision_representation(
            stage_id=stage_id,
            collision_prim_id=prim_id,
            run_asynchronously=False,  # Use synchronous request for simplicity
            on_result=self.on_convex_representation_ready
        )

        return False

    def on_convex_representation_ready(self, result, convex_hulls):
        """
        Callback function that is called when the convex hulls are ready.
        """
        # self.result = result
        # self.convex_hulls = convex_hulls

        if result == PhysxCollisionRepresentationResult.RESULT_VALID:
            # Process the convex hulls
            for convex_hull_data in convex_hulls:
                # Get vertices and polygons from convex hull data
                vertices = convex_hull_data["vertices"]
                
                # Transform vertices to world space
                vertex_world_list = []
                for vertex in vertices:
                    vert = torch.tensor(vertex, dtype=torch.float32)
                    vertex_world = self.transform_matrix.Transform(Gf.Vec3d(vert[0].item(), vert[1].item(), vert[2].item()))
                    vertex_world_list.append([vertex_world[0], vertex_world[1], vertex_world[2]])

                # Create Delaunay triangulation and mesh
                hull = Delaunay(vertex_world_list)
                mesh = trimesh.Trimesh(vertices=vertex_world_list, faces=hull.simplices)

                # Calculate the closest point and distance
                closest_point, distance, _ = trimesh.proximity.closest_point(mesh, [self.point])
                
                # Check if the distance is below threshold
                if distance <= self.threshold:
                    return True  # Collision detected
        return False
