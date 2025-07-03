#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Third Party
import torch
from typing import Dict, List

# CuRobo
from curobo.util.torch_utils import get_torch_jit_decorator
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose

# Local Folder
from .cost_base import CostBase, CostConfig

@get_torch_jit_decorator()
def compute_yaw_cost(pos_batch: torch.Tensor, target_positions: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute yaw cost to align robot's base yaw with the center of two target positions.
    
    Args:
        pos_batch: position batch [N, batch_size, dofs], where first 3 dims are base [x,y,yaw]
        target_positions: Target positions tensor [2, 3] containing two target positions
        weight: Cost weight [1]
    
    Returns:
        cost: Yaw alignment cost [N, batch_size]
    """
    # Extract base position and yaw from pos_batch
    base_pos = pos_batch[..., :2]  # [N, batch_size, 2]
    base_yaw = pos_batch[..., 2]  # [N, batch_size]

    # Calculate center point between two targets
    center_pos = torch.mean(target_positions, dim=0)  # [3]

    # Calculate desired yaw (angle from base to center point)
    delta_pos = center_pos[:2] - base_pos  # [N, batch_size, 2]
    desired_yaw = torch.atan2(delta_pos[..., 1], delta_pos[..., 0])  # [N, batch_size]

    # Calculate yaw difference and normalize to [-pi, pi]
    yaw_diff = base_yaw - desired_yaw  # [N, batch_size]
    yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))  # normalize to [-pi, pi]

    # Compute cost as squared yaw difference
    cost = weight * (yaw_diff ** 2)  # [N, batch_size]
    return cost

class YawCost(CostBase):
    """Cost function for aligning robot's yaw with target positions."""

    def __init__(self, config: CostConfig):
        """Initialize yaw cost function.
        
        Args:
            config: Cost configuration
        """
        super(YawCost, self).__init__(config)

    def forward(self, pos_batch: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        """Compute yaw alignment cost.
        
        Args:
            pos_batch: position batch [N, batch_size, dofs]
            target_pos_dict: Dictionary mapping link names to target poses
        
        Returns:
            cost: Yaw alignment cost [N, batch_size]
        """
        # Extract target positions from poses
        return compute_yaw_cost(pos_batch, target_positions, self.weight)
