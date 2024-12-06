#
import torch

class RiskModel:
    """
    A class to calculate the risk cost based on LiDAR point cloud data and UAV state.
    """

    def __init__(self, risk_map, safety_threshold=5.0):
        """
        Initialize the RiskCostCalculator.

        Args:
            risk_map (torch.Tensor): A 3D tensor representing the risk map.
            safety_threshold (float): A threshold for determining risky areas.
        """
        self.risk_map = risk_map  # 3D tensor where each value represents risk level at a map point
        self.safety_threshold = safety_threshold

    def calculate_risk_cost(self, point_cloud, uav_position):
        """
        Calculate the risk cost based on LiDAR point cloud data and UAV state.

        Args:
            point_cloud (torch.Tensor): A tensor of shape (N, 3) containing LiDAR point cloud data.
            uav_position (torch.Tensor): A tensor of shape (3,) representing the UAV's current position.

        Returns:
            float: The computed risk cost.
        """
        if not isinstance(point_cloud, torch.Tensor):
            raise ValueError("point_cloud must be a torch.Tensor")
        if not isinstance(uav_position, torch.Tensor):
            raise ValueError("uav_position must be a torch.Tensor")

        # Ensure dimensions are correct
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError("point_cloud must have shape (N, 3)")
        if uav_position.shape != (3,):
            raise ValueError("uav_position must have shape (3,)")

        # Adjust point cloud relative to the UAV position
        relative_points = point_cloud - uav_position

        # Compute indices in the risk map based on relative positions
        indices = torch.floor(relative_points).long()

        # Mask out points that are out of map bounds
        valid_mask = (indices >= 0).all(dim=1) & (indices < torch.tensor(self.risk_map.shape)).all(dim=1)
        valid_indices = indices[valid_mask]

        # Extract risk levels from the map at valid indices
        risk_levels = self.risk_map[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]

        # Calculate the total risk cost
        risk_cost = torch.sum(risk_levels[risk_levels > self.safety_threshold]).item()

        return risk_cost

    def update(self, new_risk_map):
        """
        Update the risk map.

        Args:
            new_risk_map (torch.Tensor): A new 3D tensor representing the updated risk map.
        """
        if not isinstance(new_risk_map, torch.Tensor):
            raise ValueError("new_risk_map must be a torch.Tensor")
        if new_risk_map.ndim != 3:
            raise ValueError("new_risk_map must have 3 dimensions")

        self.risk_map = new_risk_map

    def set_safety_threshold(self, new_threshold):
        """
        Set a new safety threshold.

        Args:
            new_threshold (float): The new safety threshold.
        """
        if not isinstance(new_threshold, (int, float)):
            raise ValueError("new_threshold must be a number")

        self.safety_threshold = new_threshold


    def fatality_risk(self):
        return