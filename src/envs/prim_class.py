# This script defines the prim in Isaac Sim environment


class PrimClass():
    def __init__(self, PrimMesh=None):
        self.set_mesh(PrimMesh)
        if PrimMesh:
            assert PrimMesh.IsValid() and PrimMesh.HasProperty('extent')
            self.calculate_attributes()
            self.calculate_position()

    def set_mesh(self, PrimMesh=None):
        self.mesh = PrimMesh
        self.isComplete = False

    def calculate_attributes(self):
        # Get the basic information
        self.name = self.mesh.GetName()
        self.path = self.mesh.GetPath().pathString
        # Get the categories

    def calculate_position(self):
        # Get its parent Xform of this mesh
        self.parent = self.mesh.GetParent()
        # Get the translate of its parent
        parent_path = self.parent.GetPath().pathString
        # print("prim name: ", cp_path) # Show full name
        import omni.isaac.core.utils.prims as prims_utils
        translation = prims_utils.get_prim_attribute_value(parent_path, attribute_name="xformOp:translate")
        
        # Get its outline, extent values 
        extent_attr = self.mesh.GetAttribute('extent')
        bbox = extent_attr.Get()
        print("The bounding box: ", bbox)
        # extent_attr may be None !
        if bbox:
            assert len(bbox) == 2
            self.bbox_min,self.bbox_max = bbox
            # Calculate the size
            self.size = self.bbox_max - self.bbox_min
            # Calculate the center
            local_center = list((self.bbox_max + self.bbox_min)/2)
            
            # calculate the position
            self.position = translation + local_center
            self.isComplete = True
        else:
            self.isComplete = False

    def show(self):
        print(f"The mesh path of Xform prim: {self.path}")
        print(f"The size of prim: {self.size}")
        print(f"The bounding box of prim: {self.bbox_min}, {self.bbox_max}")
        print(f"The position of prim: {self.position}")