import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


class PointPublisher(Node):
    def __init__(self, node_name, topic_name):
        super().__init__(node_name)
        self.publisher_ = self.create_publisher(PointCloud2, topic_name, 10)
        self.frame_id = "map"

    def pub(self, point_tensor):
        if point_tensor is None or point_tensor.numel() == 0:
            # self.get_logger().warn("No point cloud data available")
            return

        # Ensure the tensor is on CPU and convert to numpy array
        point_data = point_tensor.cpu().numpy()  # Shape: (N, 3)
        # self.get_logger().info(f"Publishing point cloud with {point_data.shape[0]} points")

        # Convert to ROS2 PointCloud2 
        point_cloud_msg = self.convert_to_pointcloud2(point_data)

        # Publish the PointCloud2 message
        self.publisher_.publish(point_cloud_msg)
        # self.get_logger().info("Published grid map as PointCloud2")

    def convert_to_pointcloud(self, points):
        """
        Convert N x 3 point cloud data to ROS2 PointCloud2 format
        """
        # Create header
        header = Header()
        header.frame_id = self.frame_id
        header.stamp = self.get_clock().now().to_msg()

        # Define the fields of PointCloud2
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Generate PointCloud2 messages using the sensor_msgs_py tool
        point_cloud_msg = point_cloud2.create_cloud(header, fields, points)

        return point_cloud_msg

    def convert_to_pointcloud2(self, points):
        """
        Convert N x 3 point cloud data to ROS2 PointCloud2 format
        """

        # Create header
        header = Header()
        header.stamp = rclpy.time.Time().to_msg()
        header.frame_id = self.frame_id

        # Define PointCloud2 fields
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Flatten the points for serialization
        points_serialized = points.tobytes()

        # Create PointCloud2 message
        point_cloud_msg = PointCloud2()
        point_cloud_msg.header = header
        point_cloud_msg.height = 1  # Unordered point cloud (height = 1)
        point_cloud_msg.width = points.shape[0]  # Number of points
        point_cloud_msg.fields = fields
        point_cloud_msg.is_bigendian = False  # Use little-endian
        point_cloud_msg.point_step = 12  # Each point consists of 3 float32 values (3 * 4 bytes)
        point_cloud_msg.row_step = point_cloud_msg.point_step * points.shape[0]
        point_cloud_msg.data = points_serialized
        point_cloud_msg.is_dense = True  # No invalid points

        return point_cloud_msg


if __name__ == '__main__':
    import torch
    import time

    # Define the base data for each column
    x_values = torch.linspace(4.4061, 4.4119, steps=7)  # x ranges from 4.4061 to 4.4119, constant for each row
    y_values = torch.linspace(-0.26720, 0.00055014, steps=72)  # y ranges from -0.26720 to 0.00055014
    z_values = torch.linspace(-0.33005, -0.47753, steps=6)  # z ranges from -0.33005 to -0.47753
    # Generate the grid
    x_grid, y_grid, z_grid = torch.meshgrid(x_values, y_values, z_values, indexing="ij")
    # Flatten the grid into (n, 3) format
    point_cloud_tensor = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1)

    # Initialize ROS 2
    rclpy.init(args=None)

    # Create the LidarPublisher node and pass the LiDAR instance
    lidar_publisher = PointPublisher('point_publish_node', '/points')

    while rclpy.ok():
        # Update the point cloud data in the ROS node
        lidar_publisher.pub(point_cloud_tensor)
        # Wait for the next update cycle
        time.sleep(0.1)

    # Cleanup
    lidar_publisher.destroy_node()
    rclpy.shutdown()
