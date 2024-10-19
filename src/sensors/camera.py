#
# This script create a customized camera
# based on the range_sensor  
# 

##
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Sequence, Tuple
import open3d as o3d
from pxr import Gf

import omni
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils


class DepthCamera():
    def __init__(self, 
                 config,
                 number: Optional[int] = 0,
                 frequency: Optional[int] = 20,
                 resolution: Optional[np.ndarray] = [1920, 1200],
                 translation: Optional[np.ndarray] = [0.0, 0.0, 0.05],
                 orientation: Optional[np.ndarray] = [0, 0, 0],
                 visualization: Optional[bool] = False,
    ) -> None:
        
        self.number = number
        self.update_config(config)
        
        self._camera.initialize()
        self._camera.add_motion_vectors_to_frame()

        # self.translation_in_local = translation
        # self.orientation_in_local = orientation

        self.calibration()

        # self.visualization = visualization
        return
    
    def update_config(self, config):
        '''
        config = config['camera']
        '''
        prim_path=f"/World/envs/env_{self.number}/Crazyflie/body/camera"
        self.visualization = config['visualization']
        self._camera = Camera(prim_path=prim_path,
                            frequency=config['frequency'],
                            resolution=tuple(config['resolution']),
                            translation=config['translation'],
                            orientation=rot_utils.euler_angles_to_quats(config['orientation'], degrees=True),)
        return

    def calibration(self):
        # OpenCV camera matrix and width and height of the camera sensor, from the calibration file
        width, height = 1920, 1200
        self.camera_matrix = [[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]

        # Pixel size in microns, aperture and focus distance from the camera sensor specification
        # Note: to disable the depth of field effect, set the f_stop to 0.0. This is useful for debugging.
        pixel_size = 3 * 1e-3   # in mm, 3 microns is a common pixel size for high resolution cameras
        f_stop = 1.8            # f-number, the ratio of the lens focal length to the diameter of the entrance pupil
        focus_distance = 0.6    # in meters, the distance from the camera to the object plane

        # Calculate the focal length and aperture size from the camera matrix
        ((fx,_,cx),(_,fy,cy),(_,_,_)) = self.camera_matrix
        horizontal_aperture =  pixel_size * width                   # The aperture size in mm
        vertical_aperture =  pixel_size * height
        focal_length_x  = fx * pixel_size
        focal_length_y  = fy * pixel_size
        focal_length = (focal_length_x + focal_length_y) / 2         # The focal length in mm

        # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
        self._camera.set_focal_length(focal_length / 10.0)                # Convert from mm to cm (or 1/10th of a world unit)
        self._camera.set_focus_distance(focus_distance)                   # The focus distance in meters
        self._camera.set_lens_aperture(f_stop * 100.0)                    # Convert the f-stop to Isaac Sim units
        self._camera.set_horizontal_aperture(horizontal_aperture / 10.0)  # Convert from mm to cm (or 1/10th of a world unit)
        self._camera.set_vertical_aperture(vertical_aperture / 10.0)

        self._camera.set_clipping_range(0.05, 1.0e5)

    def get_camera_image(self):
        # 3d
        # points_2d = self._camera.get_image_coords_from_world_points(
        #     np.array([self.copter.get_world_pose()[0], self.copter.get_world_pose()[0]])
        # )
        # points_3d = self._camera.get_world_points_from_image_coords(points_2d, np.array([24.94, 24.9]))
        # print("points_2d", points_2d)
        # print("points_3d: ", points_3d)

        # pointcloud_data = self._camera.get_pointcloud()
        # if pointcloud_data is not None:
        #         print("pointcloud_data shape: ", pointcloud_data.shape)

        # 2d
        image = self._camera.get_rgba()[:, :, :3]

        if self.visualization:
            imgplot = plt.imshow(image)
            plt.show()
            # if pointcloud_data is not None:
            #     point_cloud = o3d.geometry.PointCloud()
            #     point_cloud.points = o3d.utility.Vector3dVector(pointcloud_data)
            #     # visualize the point cloud
            #     o3d.visualization.draw_geometries([point_cloud])

        return
    
    def get_fov(self):
        hfov = self._camera.get_horizontal_fov()
        vfov = self._camera.get_vertical_fov()

        print("FOV: %f x %f " % (np.rad2deg(hfov), np.rad2deg(vfov)))
        return hfov, vfov
    
    def get_intrinsic_pramters(self):
        # self.camera_matrix is 3 x 3 in inhomogeneous coordinates
        camera_matrix_np = np.array(self.camera_matrix)
        zero_column = np.zeros((camera_matrix_np.shape[0], 1))
        intrinsic_paramter_matrix = np.hstack((camera_matrix_np, zero_column))
        return intrinsic_paramter_matrix
    
    def get_extrinsic_parameters(self, position, orientation):
        """
        Input:
        position: numpy array containing position (x, y, z)
        orientation: numpy array containing Euler angles (phi, omega, theta), 
        representing rotation around the x, y, z axes respectively (in radians)

        Output:
        extrinsic_paramter_matrix: 4x4 matrix combining rotation and translation
        """
        x, y, z = position
        phi, omega, theta = orientation

        # Rotation matrix around x-axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        
        # Rotation matrix around y-axis
        Ry = np.array([
            [np.cos(omega), 0, np.sin(omega)],
            [0, 1, 0],
            [-np.sin(omega), 0, np.cos(omega)]
        ])
        
        # Rotation matrix around z-axis
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        # Multiply the three rotation matrices in the order Rz * Ry * Rx
        R = np.dot(np.dot(Rz, Ry), Rx)

        # Translation vector, representing the position (x, y, z)
        T = np.array([x, y, z])

        # Combine rotation matrix R and translation vector T into a 4x4 transformation matrix
        # Create an identity matrix for the 4x4 format
        extrinsic_paramter_matrix = np.eye(4)
        
        # Set the upper left 3x3 block to the rotation matrix R
        extrinsic_paramter_matrix[:3, :3] = R
        
        # Set the upper right 3x1 block to the translation vector (as a column)
        extrinsic_paramter_matrix[:3, 3] = T

        return extrinsic_paramter_matrix
    
