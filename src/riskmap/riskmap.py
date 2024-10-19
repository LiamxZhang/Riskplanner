# This code can 
# transfer the omniverse map to the point cloud
# then compute the risk map corresponding to each map grid
# grid map
# build the 

import datetime
import time
import math
import gym
import hydra
import torch
import numpy as np

import rclpy
from rclpy.clock import Clock
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField 
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32

import omni
from omni.isaac.kit import SimulationApp
from omni.isaac.occupancy_map import _occupancy_map
import omni.isaac.core.utils.prims as prims_utils


# ROS2 publisher for point cloud data
class PointPublisher(Node):
    def __init__(self):
        super().__init__('point_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, '/isaacsim/map/point_cloud', 10)
        self.publishing_flag = True  # initialized as True for publishing

    def publish_data(self, cloud_data):
        # the imported cloud_data is a N*3 Numpy array
        head = Header()
        head.stamp = Clock().now().to_msg()
        head.frame_id = "map"  # rviz2 map coordinate
        msg = create_cloud_xyz32(head, cloud_data)
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: %s' % str(cloud_data))


class RiskMap():
    def __init__(self, env=None): 
        # import the built SimulationApp -- env
        # if the env is not created, create the env from the imported map
        self.env = env
        
        # initialize the point cloud publisher
        rclpy.init(args=None)
        self.point_cloud_publisher = PointPublisher()

        self.cellsize = 0.05
        self.minbound = np.array([-2, -2, -1])
        self.maxbound = np.array([2, 2, 1])
        # map dim
        self.map_dim = np.floor((self.maxbound-self.minbound)/self.cellsize)
        # print("map dimension: ", self.map_dim)

    def get_point_cloud(self, cell_size=.05, origin=(0, 0, 0), minbound=(-2, -2, -1), maxbound=(2, 2, 1)):
        physx = omni.physx.acquire_physx_interface()
        stage_id = omni.usd.get_context().get_stage_id()
        obs_generator = _occupancy_map.Generator(physx, stage_id)
        # 0.05m cell size, output buffer will have 4 for obstacle occupied cells, 
        # 5 for unoccupied empty, and 6 for (unknown) cells that cannot be seen
        # this assumes your usd stage units are in m, and not cm
        obs_generator.update_settings(cell_size, 4, 5, 6)
        # Set location to map from and the min and max bounds to map to
        # arg0: origin, arg1: min bound, arg2: max bound
        obs_generator.set_transform(origin, minbound, maxbound)
        obs_generator.generate3d()
        # Get locations of the occupied cells in the stage
        self.obstacle_points = obs_generator.get_occupied_positions()
        # print("Points:", self.obstacle_points)

        # Get computed 2d occupancy buffer
        # buffer = self.obs_generator.get_buffer()

        # Get dimensions for 2d buffer
        # dims = self.obs_generator.get_dimensions()

        return self.obstacle_points
    

    # tramsform the list of carb.float3 data into a N*3 Numpy array
    def carb2array(self, points):
        cloud_data = np.empty(shape=(0,3))
        for point in points:
            point = np.array(point).reshape(1, -1)
            cloud_data = np.append(cloud_data, point, axis=0)
        # print("Point array: ", cloud_data)

        return cloud_data

    # conduct the ROS2 publish function
    def pub(self, args=None):
        cloud_data = self.carb2array(self.obstacle_points)
        # print("Point cloud: ", cloud_data)
        self.point_cloud_publisher.publish_data(cloud_data)
        # rclpy.sleep(1) 

    def pub_end(self):
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        self.point_cloud_publisher.destroy_node()
        rclpy.shutdown()

    # get all children prim name and their positions
    def get_prim_pos(self):
        predicate = lambda path: True
        child_prims = prims_utils.get_all_matching_child_prims("/", predicate)
        # print("child_prim: ", child_prims)
        for prim in child_prims:
            pos = prim.GetAttribute("xformOp:translate").Get()
            
            if pos is not None:
                print("child_prim: {}, position: {}".format(prim, pos))
            else:
                # attribute_names = prims_utils.get_prim_attribute_names(prim)
                pass

    # whole world as a point cloud
    def build_point_cloud(self):
        # new a map matrix
        self.map_data = np.zeros(tuple(self.map_dim.astype(np.int64)))
        # if obstacle, value is 1

        # match the coordinate with the map matrix
        return self.map_data

    def coord2map(self):
        pass
    

    def map2coord(self):
        pass


    def pub_risk_map(self):
        # import the map (numpy array N*3), output the risk map message
        #
        # points_position = self.map_data
        points_position = self.carb2array(self.obstacle_points)
        num_temp = np.shape(points_position)[0]
        points_intensity = np.ones(num_temp) # risk probability
        points_time = np.zeros(num_temp)

        risk_map_msg = PointCloud2()
        risk_map_msg.height = 1
        risk_map_msg.width = len(points_position)
        risk_map_msg.header.frame_id = 'map'
        risk_map_msg.header.stamp = Clock().now().to_msg()
        risk_map_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="ring", offset=12, datatype=PointField.UINT16, count=1),
            PointField(name="intensity", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="time", offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgba", offset=24, datatype=PointField.UINT32, count=1)
        ]
        risk_map_msg.is_bigendian = False # big endian
        risk_map_msg.point_step = 24 
        risk_map_msg.row_step = risk_map_msg.point_step * risk_map_msg.width
        risk_map_msg.is_dense = True

        # 
        points_ring = np.zeros(points_position.shape[0], dtype=int)
        NumberOfChannels = 64
        VerticalFOVUpper = 40.0
        VerticalFOVLower = -40.0
        ang_res = (VerticalFOVUpper - VerticalFOVLower) / float(NumberOfChannels)

        msg_data = []
        for i in range(points_position.shape[0]):
            x = points_position[i][0]
            y = points_position[i][1]
            z = points_position[i][2]
            verticalAngle = math.atan2(z, math.sqrt(x*x + y*y)) * 180 / math.pi
            points_ring[i] = (verticalAngle - VerticalFOVLower) // ang_res

            msg_data.append([points_position[i][0], points_position[i][1], points_position[i][2], 
                                    points_ring[i], points_intensity[i], points_time[i]])
                    
        risk_map_msg.data = np.array(msg_data).astype(np.float32).tobytes()

        # print("Point cloud: ", msg_data)

        self.point_cloud_publisher.publisher_.publish(risk_map_msg)
# if __name__ == "__main__":
