from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_rsp_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("episode1_with_gripper", package_name="episode1_urdf_with_gripper_moveit").to_moveit_configs()
    return generate_rsp_launch(moveit_config)
