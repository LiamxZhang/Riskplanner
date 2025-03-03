#!/usr/bin/env python
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MultiModalFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, cnn_output_dim=64, mlp_output_dim=64):
        # Initialize parent class
        super().__init__(observation_space, cnn_output_dim + mlp_output_dim)

        # output_dim = (input_dim + 2*padding - kernel_size)/stride + 1
        length, width, height = self.calculate_conv3d_output_shape(observation_space.spaces["gridmap"].shape[1:], kernel_size=3, stride=2, padding=1)
        length, width, height = self.calculate_conv3d_output_shape((length, width, height), kernel_size=3, stride=2, padding=1)
        flattened_size = 64 * length * width * height  # 64 is the out_channels of the last Conv3d layer
        
        def init_weights(m):
            if isinstance(m, nn.Conv3d):
                # Kaiming/He initialization for Conv3d layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Linear):
                # Xavier/Glorot initialization for Linear layers
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)

        # Exract the feature of grid_map via 3D CNN
        self.cnn = nn.Sequential(
            # [batch_size, 1, 32, 32, 32]
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            # [batch_size, 32, 16, 16, 16]
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            # [batch_size, 64, 8, 8, 8]
            nn.ReLU(),
            nn.Flatten(),
            # [batch_size, 64*8*8*8]
            nn.Linear(flattened_size, cnn_output_dim),  
            # [batch_size, cnn_output_dim]
            nn.ReLU()
        )
        # Apply weight initialization
        self.cnn.apply(init_weights)

        # 自动获取无人机状态维度
        state_shape = observation_space.spaces["state"].shape
        state_dim = state_shape[0] if len(state_shape) > 0 else 1  # 处理标量情形

        # MLP to process the drone state
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 128),  # 动态输入维度
            nn.ReLU(),
            nn.Linear(128, mlp_output_dim),
            nn.ReLU()
        )
        # Apply weight initialization
        self.mlp.apply(init_weights)

    def forward(self, observations):
        # Unsqueeze the grid_map [batch_size, 1, length, width, height]
        grid_map = observations["gridmap"] 
        # Input dimension: [batch_size, 1, length, width, height]
        grid_map_features = self.cnn(grid_map)

        # Process the drone state
        state_features = self.mlp(observations["state"])

        # Merge 2 features
        return torch.cat([grid_map_features, state_features], dim=1)

    # Function to calculate Conv3d output shape
    def calculate_conv3d_output_shape(self, input_shape, kernel_size, stride, padding):
        # Input observation_space shape: [batch_size, length, width, height] 
        x, y, z = input_shape
        output_x = (x + 2 * padding - kernel_size) // stride + 1
        output_y = (y + 2 * padding - kernel_size) // stride + 1
        output_z = (z + 2 * padding - kernel_size) // stride + 1
        return output_x, output_y, output_z