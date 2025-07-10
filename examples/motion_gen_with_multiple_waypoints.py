# Third-party
import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenResult,
    PoseCostMetric,
)
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml

# Define a simple world with a table
world_config = {
    "cuboid": {
        "table": {
            "dims": [5.0, 5.0, 0.1],
            "pose": [100.0, 0.0, -0.1, 1, 0, 0, 0],
        },
    },
}

# Load MotionGen configuration
motion_gen_config = MotionGenConfig.load_from_robot_config(
    "franka.yml",
    world_config,
    finetune_trajopt_iters=2000,
    trajopt_tsteps=512,
    maximum_trajectory_dt=2.0,
    maximum_trajectory_time=20,
)
motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()

# Define offset waypoints
start_joint_positions = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]
goal_pose = Pose.from_list([0.505, 0.065, 0.22, 1.0, 0.0, 0.0, 0.0])
offset_positions = torch.tensor([[0.05, 0.05, -0.2], [0.05, 0.05, -0.1]], dtype=torch.float32).cuda()
offset_rotations = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32).cuda()
tstep_fraction = [0.5, 0.25]

pose_metric = PoseCostMetric.create_offset_waypoint_metric(
    offset_position=offset_positions,
    offset_rotation=offset_rotations,
    tstep_fraction=tstep_fraction,
)

start_state = JointState.from_position(
    torch.tensor([start_joint_positions], dtype=torch.float32).cuda(),
    joint_names=[
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7"
    ],
)

# Plan trajectory
result = motion_gen.plan_single(
    start_state,
    goal_pose,
    MotionGenPlanConfig(
        max_attempts=1,
        pose_cost_metric=pose_metric,
        enable_graph=True,
    ),
)

if result.success:
    print("Trajectory generation succeeded.")

    traj = result.optimized_plan
    tensor_args = TensorDeviceType()

    # Load robot configuration for FK computation
    config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))
    urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
    robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
    kin_model = CudaRobotModel(robot_cfg.kinematics)

    # Run forward kinematics
    joint_positions = traj.position
    fk_results = kin_model.get_state(joint_positions)
    ee_positions = fk_results.ee_position
    ee_quaternions = fk_results.ee_quaternion

    goal_position = goal_pose.position
    offsets = torch.tensor(offset_positions, device=ee_positions.device)
    expected_positions = goal_position - offsets

    for i, (expected_pos, frac) in enumerate(zip(expected_positions, tstep_fraction)):
        expected_step = int(len(joint_positions) * (1.0 - frac))
        print(f"\nOffset Waypoint {i}:")
        print(f"Expected position: {expected_pos.tolist()}")
        print(f"Expected timestep ({frac*100:.1f}%): {expected_step}")

        for j in range(max(0, expected_step - 2), min(len(ee_positions), expected_step + 3)):
            dist = torch.norm(ee_positions[j] - expected_pos).item()
            print(f"Timestep {j}: distance = {dist:.4f}")

        dists = torch.norm(ee_positions - expected_pos.unsqueeze(0), dim=1)
        closest_step = torch.argmin(dists).item()
        closest_dist = dists[closest_step].item()
        print(f"\nClosest approach to offset waypoint {i}:")
        print(f"Timestep: {closest_step}, Distance: {closest_dist:.4f}")

else:
    print("Trajectory generation failed.")
    print("Status:", result.status)