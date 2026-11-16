#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
import numpy as np
import time
import os
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_msgs.msg import CollisionObject, Constraints, OrientationConstraint
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped, Pose
from robot_arm_interfaces.srv import RobotEntries
import tf_transformations
from typing import Dict, List


class MoveitAPI(Node):
    def __init__(self):
        super().__init__('moveit_api_node')
        self.declare_parameter('robot_name', 'episode1_urdf_1113')
        self.declare_parameter('package_name', 'episode1_urdf_student_moveit')
        self.declare_parameter('planning_group', 'episode_arm')
        self.declare_parameter('planning_pipeline', 'ompl')
        self.declare_parameter('planning_attempts', 3)
        self.declare_parameter('max_velocity_scaling', 0.7)
        self.declare_parameter('max_acceleration_scaling', 0.3)
        
        robot_name = self.get_parameter('robot_name').value
        package_name = self.get_parameter('package_name').value
        self.planning_group = self.get_parameter('planning_group').value
        planning_pipeline = self.get_parameter('planning_pipeline').value
        planning_attempts = self.get_parameter('planning_attempts').value
        max_velocity_scaling = self.get_parameter('max_velocity_scaling').value
        max_acceleration_scaling = self.get_parameter('max_acceleration_scaling').value
        
        self.get_logger().info(f"Initializing {robot_name} MoveAPI")
        
        self.robot_busy = False
        self.service = self.create_service(
            RobotEntries,
            'Moveit_Robo_Node',
            self._task_entry_cb,
        )
        
        robot_config = self._build_moveit_config(
            robot_name=robot_name, package_name=package_name,
            planning_pipeline=planning_pipeline,
            planning_attempts=planning_attempts,
            max_velocity_scaling=max_velocity_scaling,
            max_acceleration_scaling=max_acceleration_scaling
            )
        
        try:
            self.robot = MoveItPy(node_name="moveit_py", config_dict=robot_config)
            self.arm = self.robot.get_planning_component(self.planning_group)
            self.planning_scene_monitor = self.robot.get_planning_scene_monitor()
            
            self.get_logger().info("Waiting for robot status...")
            time.sleep(1.0)
            self.get_logger().info("MoveIt Initialized")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Moveit: {e}")
            raise
        
    def _build_moveit_config(self, robot_name, package_name, planning_pipeline,
                                  planning_attempts, max_velocity_scaling,
                                  max_acceleration_scaling) -> Dict:
        """Constrict Moveit Config into Dict"""
        base_config = MoveItConfigsBuilder(
            robot_name=robot_name, package_name=package_name
        ).robot_description_semantic(
            "config/episode1_urdf_1113.srdf", {"name": robot_name}
        ).planning_pipelines(
            default_planning_pipeline=planning_pipeline
        ).to_dict()

        # 添加 per-group OMPL 覆盖 (默认规划器、投影评估器) 
        if "ompl" in base_config:
            base_config["ompl"][self.planning_group] = {
                "default_planner_config": "RRTConnect",
                "planner_configs": ["RRTConnect", "RRT", "PRM", "EST", "BiEST"],
                "projection_evaluator": "joints(joint1,joint2,joint3)",
            }

        config = {
            **base_config,
            "planning_scene_monitor": {
                "name": "planning_scene_monitor",
                "robot_description": "robot_description",
                "joint_state_topic": "/joint_states",
                "attached_collision_object_topic": "/moveit_cpp/planning_scene_monitor",
                "publish_planning_scene_topic": "/moveit_cpp/publish_planning_scene",
                "monitored_planning_scene_topic": "/moveit_cpp/monitored_planning_scene",
                "wait_for_initial_state_timeout": 10.0,
            },
            "planning_pipelines": {"pipeline_names": [planning_pipeline]},
            "plan_request_params": {
                "planning_attempts": planning_attempts,
                "planning_pipeline": planning_pipeline,
                "max_velocity_scaling_factor": max_velocity_scaling,
                "max_acceleration_scaling_factor": max_acceleration_scaling,
            },
            "trajectory_execution": {"allowed_start_tolerance": 0.05},
        }
        return config
    
    def move_to_target(self, target_name:str) -> bool:
        """Plan and Execute Goal State (E.g "home"、"test1") """
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(configuration_name=target_name)
        plan_result = self.arm.plan()
        if plan_result:
            self.robot.execute(group_name=self.planning_group, robot_trajectory=plan_result.trajectory)
            time.sleep(0.5)
            return True
        self.get_logger().warn(f"Failed to plan Goal State: '{target_name}'")
        return False
    
    def move_to_degrees(self, angles_deg: List[float]) -> bool:
        """Plan and execute joint angle (motor angle，degree) """
        offsets = [180, 90, 83, 30, 110, 30]
        joint_angles_rad = [np.deg2rad(angles_deg[i] - offsets[i]) for i in range(6)]
        joint_target = {f"joint{i+1}": joint_angles_rad[i] for i in range(6)}

        robot_state = RobotState(self.robot.get_robot_model())
        robot_state.joint_positions = joint_target

        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(robot_state=robot_state)
        plan_result = self.arm.plan()
        if plan_result:
            self.robot.execute(group_name=self.planning_group, robot_trajectory=plan_result.trajectory)
            time.sleep(0.5)
            return True
        self.get_logger().warn(f"Failed to plan Joint Angle '{angles_deg}'")
        return False
    
    def move_to_pose(self, x, y, z, x_rot, y_rot, z_rot) -> bool:
        """Plan and execute the pose (position:mm, rotation:rad) """
        q_x, q_y, q_z, q_w = tf_transformations.quaternion_from_euler(
            x_rot, y_rot, z_rot, axes='sxyz'
        )
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose.position.x = x / 1000.0
        pose_goal.pose.position.y = y / 1000.0
        pose_goal.pose.position.z = z / 1000.0
        pose_goal.pose.orientation.x = q_x
        pose_goal.pose.orientation.y = q_y
        pose_goal.pose.orientation.z = q_z
        pose_goal.pose.orientation.w = q_w
        
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link='link6')
        plan_result = self.arm.plan()
        if plan_result:
            self.robot.execute(group_name=self.planning_group, robot_trajectory=plan_result.trajectory)
            time.sleep(0.5)
            return True
        self.get_logger().warn(f"Failed to plan Pose [{x:.1f}, {y:.1f}, {z:.1f}, {x_rot:.1f}, {y_rot:.1f}, {z_rot:.1f}]")
        return False
    
    # ==================== Scene Management ====================
    
    def add_collision_box(self, box_id, position, dimensions):
        """向规划场景添加碰撞箱体 (position/dimensions 单位: 米) """
        with self.planning_scene_monitor.read_write() as scene:
            collision_object = CollisionObject()
            collision_object.header.frame_id = "base_link"
            collision_object.id = box_id
            box_pose = Pose()
            box_pose.position.x, box_pose.position.y, box_pose.position.z = position
            box_pose.orientation.w = 1.0
            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = dimensions
            collision_object.primitives.append(box)
            collision_object.primitive_poses.append(box_pose)
            collision_object.operation = CollisionObject.ADD
            scene.apply_collision_object(collision_object)
            scene.current_state.update()
        self.get_logger().info(f"已添加碰撞箱体 '{box_id}'")

    def remove_collision_object(self, object_id):
        """移除所有碰撞物体"""
        with self.planning_scene_monitor.read_write() as scene:
            scene.remove_all_collision_objects()
            scene.current_state.update()
        self.get_logger().info("已移除所有碰撞物体")

    def check_pose_collision(self, position, rotation) -> bool:
        """检查指定位姿是否碰撞 (position:mm, rotation:rad) """
        with self.planning_scene_monitor.read_only() as scene:
            robot_state = scene.current_state
            original_joint_positions = robot_state.get_joint_group_positions(self.planning_group)

            pose_goal = Pose()
            pose_goal.position.x = position[0] / 1000.0
            pose_goal.position.y = position[1] / 1000.0
            pose_goal.position.z = position[2] / 1000.0
            q_x, q_y, q_z, q_w = tf_transformations.quaternion_from_euler(*rotation, axes='sxyz')
            pose_goal.orientation.x, pose_goal.orientation.y = q_x, q_y
            pose_goal.orientation.z, pose_goal.orientation.w = q_z, q_w

            robot_state.set_from_ik(self.planning_group, pose_goal, "link6")
            robot_state.update()

            is_colliding = scene.is_state_colliding(
                robot_state=robot_state,
                joint_model_group_name=self.planning_group,
                verbose=True,
            )

            # 恢复原始状态
            robot_state.set_joint_group_positions(self.planning_group, original_joint_positions)
            robot_state.update()
            return is_colliding
        
    # ==================== Status query ====================
    
    def get_current_degrees(self) -> List[float]:
        """Get instant Joint Angle (motor angle, degree) """
        with self.planning_scene_monitor.read_only() as scene:
            robot_state = scene.current_state
            joint_positions = robot_state.get_joint_group_positions(self.planning_group)
            offsets = [180, 90, 83, 30, 110, 30]
            return [np.rad2deg(joint_positions[i]) + offsets[i] for i in range(6)]
    
    def get_current_pose(self) -> Dict:
        """Get the instant pose of link6 (position:mm, rotation:rad/deg) """
        with self.planning_scene_monitor.read_only() as scene:
            robot_state = scene.current_state
            robot_state.update()
            transform_matrix = robot_state.get_frame_transform("link6")
            position = [transform_matrix[i, 3] * 1000.0 for i in range(3)]
            quaternion = tf_transformations.quaternion_from_matrix(transform_matrix)
            euler_rad = tf_transformations.euler_from_quaternion(quaternion, axes='sxyz')
            return {
                'position': position,
                'rotation_rad': list(euler_rad),
                'rotation_deg': [np.rad2deg(a) for a in euler_rad]
            }
            
    def _task_entry_cb(self, request, response):
        if self.robot_busy:
            response.success = False
            response.message = "Robot busy"
            return response

        if not request.poses:
            response.success = False
            response.message = "No poses in request"
            return response

        self.robot_busy = True
        try:
            for i, pose in enumerate(request.poses):
                x = pose.position.x * 1000.0   # m → mm (adjust if your move_to_pose already expects meters)
                y = pose.position.y * 1000.0
                z = pose.position.z * 1000.0

                q = pose.orientation
                x_rot, y_rot, z_rot = tf_transformations.euler_from_quaternion(
                    [q.x, q.y, q.z, q.w], axes='sxyz'
                )

                self.get_logger().info(f"Trying grasp {i+1}/{len(request.poses)}")
                if self.move_to_pose(x, y, z, x_rot, y_rot, z_rot):
                    response.success = True
                    response.message = f"Executed grasp {i+1}"
                    return response

            response.success = False
            response.message = "All grasp poses failed to plan/execute"
        except Exception as e:
            self.get_logger().error(f"Service error: {e}")
            response.success = False
            response.message = f"error: {e}"
        finally:
            self.robot_busy = False
        return response   
            
            
    def task(self):
        result = self.move_to_degrees(angles_deg=[36.0, 54.0, 2.0, 10.0, 24.0, 111.0])
        self.get_logger().info(f"result: '{result}'")
        

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MoveitAPI()
        # node.task()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"error: {e}")
    finally:
        os._exit(0)

if __name__ == "__main__":
    main()
