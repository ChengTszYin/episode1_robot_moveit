#!/usr/bin/env python3
"""
MoveIt Python API 演示节点

演示内容：
  演示 1: 命名配置（move_to_target）
  演示 2: 关节空间目标（move_to_degrees）
  演示 3: 位姿目标（move_to_pose）
  演示 4: 规划场景与避障
  演示 5: 获取当前状态
  演示 6: 关节约束目标（construct_joint_constraint）
  演示 7: 链接约束目标（construct_link_constraint）
  演示 8: 路径约束 — 搬水杯场景
"""

import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy, PlanningComponent
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.core.robot_state import RobotState
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import CollisionObject, Constraints, OrientationConstraint
from shape_msgs.msg import SolidPrimitive
import numpy as np
import time
import os
from typing import Dict, List
import tf_transformations
import random


class MoveItAPIDemo(Node):
    """MoveIt Python API 演示节点"""

    def __init__(self):
        """初始化节点和 MoveIt 核心三件套"""
        super().__init__('moveit_api_demo_node')

        # ==================== 参数声明 ====================
        self.declare_parameter('robot_name', 'episode1_urdf_1113')
        self.declare_parameter('package_name', 'episode1_urdf_student_moveit')
        self.declare_parameter('planning_group', 'episode_arm')
        self.declare_parameter('planning_pipeline', 'ompl')
        self.declare_parameter('planning_attempts', 3)
        self.declare_parameter('max_velocity_scaling', 0.7)
        self.declare_parameter('max_acceleration_scaling', 0.3)
        self.declare_parameter('demo_group', 0)  # 0=全部, 1-8=指定演示, 71=7a, 72=7b, 73=7c

        # ==================== 获取参数 ====================
        robot_name = self.get_parameter('robot_name').value
        package_name = self.get_parameter('package_name').value
        self.planning_group = self.get_parameter('planning_group').value
        planning_pipeline = self.get_parameter('planning_pipeline').value
        planning_attempts = self.get_parameter('planning_attempts').value
        max_velocity_scaling = self.get_parameter('max_velocity_scaling').value
        max_acceleration_scaling = self.get_parameter('max_acceleration_scaling').value
        demo_group_int = self.get_parameter('demo_group').value
        demo_map = {0: 'all', 71: '7a', 72: '7b', 73: '7c'}
        self.demo_group = demo_map.get(demo_group_int, str(demo_group_int))

        self.get_logger().info(f"正在初始化 {robot_name} 的 MoveIt API 演示")

        # ==================== 构建配置 ====================
        robot_config = self._build_moveit_config(
            robot_name, package_name, planning_pipeline,
            planning_attempts, max_velocity_scaling, max_acceleration_scaling
        )

        # ==================== 初始化 MoveIt 核心三件套 ====================
        try:
            self.robot = MoveItPy(node_name="moveit_py", config_dict=robot_config)
            self.arm: PlanningComponent = self.robot.get_planning_component(self.planning_group)
            self.planning_scene_monitor = self.robot.get_planning_scene_monitor()

            self.get_logger().info("等待机器人状态...")
            time.sleep(1.0)
            self.get_logger().info("MoveIt 初始化成功")
        except Exception as e:
            self.get_logger().error(f"MoveIt 初始化失败: {e}")
            raise

    def _build_moveit_config(self, robot_name, package_name, planning_pipeline,
                              planning_attempts, max_velocity_scaling,
                              max_acceleration_scaling) -> Dict:
        """构建 MoveIt 配置字典"""
        base_config = MoveItConfigsBuilder(
            robot_name=robot_name, package_name=package_name
        ).robot_description_semantic(
            "config/episode1_urdf_1113.srdf", {"name": robot_name}
        ).planning_pipelines(
            default_planning_pipeline=planning_pipeline
        ).to_dict()

        # 添加 per-group OMPL 覆盖（默认规划器、投影评估器）
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

    # ==================== 运动控制 ====================

    def move_to_target(self, target_name: str) -> bool:
        """规划并执行到命名配置（如 "home"、"test1"）"""
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(configuration_name=target_name)
        plan_result = self.arm.plan()
        if plan_result:
            self.robot.execute(group_name=self.planning_group, robot_trajectory=plan_result.trajectory)
            time.sleep(0.5)
            return True
        self.get_logger().warn(f"规划到 '{target_name}' 失败")
        return False

    def move_to_degrees(self, angles_deg: List[float]) -> bool:
        """规划并执行到指定关节角度（电机角度，度）"""
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
        self.get_logger().warn(f"规划到关节角度 {angles_deg} 失败")
        return False

    def move_to_pose(self, x, y, z, x_rot, y_rot, z_rot) -> bool:
        """规划并执行到指定位姿（位置:mm, 旋转:rad）"""
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
        self.arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="link6")
        plan_result = self.arm.plan()
        if plan_result:
            self.robot.execute(group_name=self.planning_group, robot_trajectory=plan_result.trajectory)
            time.sleep(0.5)
            self.get_logger().info("位姿目标运动成功")
            return True
        self.get_logger().warn(f"位姿目标 [{x:.1f}, {y:.1f}, {z:.1f}]mm 规划失败")
        return False

    # ==================== 场景管理 ====================

    def add_collision_box(self, box_id, position, dimensions):
        """向规划场景添加碰撞箱体（position/dimensions 单位: 米）"""
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
        """检查指定位姿是否碰撞（position:mm, rotation:rad）"""
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

    # ==================== 状态查询 ====================

    def get_current_degrees(self) -> List[float]:
        """获取当前关节角度（电机角度，度）"""
        with self.planning_scene_monitor.read_only() as scene:
            robot_state = scene.current_state
            joint_positions = robot_state.get_joint_group_positions(self.planning_group)
            offsets = [180, 90, 83, 30, 110, 30]
            return [np.rad2deg(joint_positions[i]) + offsets[i] for i in range(6)]

    def get_current_pose(self) -> Dict:
        """获取当前 link6 位姿（位置:mm, 旋转:rad/deg）"""
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
                'rotation_deg': [np.rad2deg(a) for a in euler_rad],
            }

    # ==================== 演示 1-5（基础功能） ====================

    def run_demo_1(self):
        """演示 1: 命名配置"""
        self.get_logger().info("=== 演示 1: 命名配置 ===")
        self.move_to_target("home")
        time.sleep(1.0)
        self.move_to_target("t1")
        time.sleep(1.0)
        self.move_to_target("home")

    def run_demo_2(self):
        """演示 2: 关节空间目标"""
        self.get_logger().info("=== 演示 2: 关节空间目标 ===")
        self.move_to_target("home")
        degree_list = [
            [55.43, 137.77, 98.84, 73.26, 89.22, 86.58],
            [137.22, 98.41, 108.23, 125.02, 39.80, 155.00],
            [180.00, 90.00, 83.00, 30.00, 110.00, 30.00],
            [252.12, 119.16, 58.00, 78.40, 176.26, 156.51],
            [338.00, 143.12, 77.44, 154.99, 208.63, 154.99],
            [180.00, 90.00, 83.00, 30.00, 110.00, 30.00],
        ]
        for angles in degree_list:
            self.move_to_degrees(angles)
            time.sleep(0.5)

    def run_demo_3(self):
        """演示 3: 位姿目标（随机位姿）"""
        self.get_logger().info("=== 演示 3: 位姿目标 ===")
        self.move_to_target("home")
        for i in range(10):
            position = [
                260.0 + random.uniform(-100.0, 100.0),
                0.0 + random.uniform(-150.0, 150.0),
                300.0 + random.uniform(-100.0, 100.0),
            ]
            rotation_deg = [
                180.0 + random.uniform(-30.0, 30.0),
                0.0 + random.uniform(-25.0, 25.0),
                90.0 + random.uniform(-30.0, 30.0),
            ]
            rotation_rad = [np.deg2rad(a) for a in rotation_deg]

            self.get_logger().info(
                f"尝试位姿 {i+1}: 位置=[{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}]mm, "
                f"旋转=[{rotation_deg[0]:.1f}, {rotation_deg[1]:.1f}, {rotation_deg[2]:.1f}]°"
            )
            success = self.move_to_pose(
                position[0], position[1], position[2],
                rotation_rad[0], rotation_rad[1], rotation_rad[2],
            )
            if success:
                self.get_logger().info(f"位姿 {i+1} 运动成功")
                time.sleep(1.0)
            else:
                self.get_logger().warn(f"位姿 {i+1} 运动失败，跳过")
                time.sleep(0.5)

        # move to home
        self.move_to_target("home")
        time.sleep(1.0)

    def run_demo_4(self):
        """演示 4: 规划场景与避障"""
        self.get_logger().info("=== 演示 4: 规划场景与避障 ===")
        self.move_to_target("home")

        self.move_to_pose(300.0, 200.0, 200.0, np.pi, 0.0, np.pi / 2)
        time.sleep(1.0)

        self.get_logger().info("添加碰撞箱体到规划场景...")
        self.add_collision_box("obstacle_box", [0.4, 0.0, 0.2], [0.05, 0.05, 0.05])
        time.sleep(1.0)

        is_colliding = self.check_pose_collision(
            [300.0, 200.0, 200.0], [np.pi, 0.0, np.pi / 2]
        )
        self.get_logger().info(f"当前位姿碰撞状态: {is_colliding}")
        time.sleep(2.0)

        self.get_logger().info("规划绕过障碍物的运动...")
        success = self.move_to_pose(350.0, -200.0, 200.0, np.pi, 0.0, np.pi / 2)
        if success:
            self.get_logger().info("成功绕过障碍物到达目标位置")
            time.sleep(2.0)

        self.remove_collision_object("obstacle_box")
        time.sleep(1.0)

    def run_demo_5(self):
        """演示 5: 获取当前状态"""
        self.get_logger().info("=== 演示 5: 获取当前状态 ===")
        self.move_to_target("home")

        test_positions = [
            [180.00, 90.00, 83.00, 30.00, 110.00, 30.00],
            [55.43, 137.77, 98.84, 73.26, 89.22, 86.58],
            [137.22, 98.41, 108.23, 125.02, 39.80, 155.00],
        ]
        for i, degrees in enumerate(test_positions, 1):
            self.get_logger().info(f"\n--- 测试位置 {i} ---")
            success = self.move_to_degrees(degrees)
            if success:
                time.sleep(0.5)
                current_degrees = self.get_current_degrees()
                self.get_logger().info("当前关节角度（度）:")
                for j, angle in enumerate(current_degrees, 1):
                    self.get_logger().info(f"  关节{j}: {angle:.2f}°")

                current_pose = self.get_current_pose()
                self.get_logger().info("当前末端位姿:")
                self.get_logger().info(
                    f"  位置(mm): x={current_pose['position'][0]:.2f}, "
                    f"y={current_pose['position'][1]:.2f}, z={current_pose['position'][2]:.2f}"
                )
                self.get_logger().info(
                    f"  旋转(度): x={current_pose['rotation_deg'][0]:.2f}°, "
                    f"y={current_pose['rotation_deg'][1]:.2f}°, z={current_pose['rotation_deg'][2]:.2f}°"
                )
                time.sleep(2.0)
            else:
                self.get_logger().warn(f"测试位置 {i} 运动失败")

        self.move_to_target("home")

    # ==================== 演示 6-8（运动约束） ====================

    def run_demo_6(self):
        """演示 6: 关节约束目标（三种容差对比）"""
        from moveit.core.kinematic_constraints import construct_joint_constraint

        self.get_logger().info("=== 演示 6: 关节约束目标（三种容差对比） ===")

        joint_values = {
            "joint1": 0.5, "joint2": -0.3, "joint3": 0.4,
            "joint4": -0.5, "joint5": 0.2, "joint6": 0.0,
        }
        joint_model_group = self.robot.get_robot_model().get_joint_model_group(
            self.planning_group
        )

        # 三种容差：严格 → 中等 → 宽松
        tolerances = [0.01, 0.05, 0.1]
        labels = ["严格(±0.57°)", "中等(±2.9°)", "宽松(±5.7°)"]
        self.move_to_target("home")
        time.sleep(1.0)
        
        for tol, label in zip(tolerances, labels):
            self.get_logger().info(f"\n--- 容差对比: {label}, tolerance={tol} rad ---")
            

            robot_state = RobotState(self.robot.get_robot_model())
            robot_state.joint_positions = joint_values

            joint_constraint = construct_joint_constraint(
                robot_state=robot_state,
                joint_model_group=joint_model_group,
                tolerance=tol,
            )

            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(motion_plan_constraints=[joint_constraint])
            plan_result = self.arm.plan()
            if plan_result:
                self.robot.execute(group_name=self.planning_group, robot_trajectory=plan_result.trajectory)
                time.sleep(0.5)
                current = self.get_current_degrees()
                self.get_logger().info(f"容差 {label} 运动成功，当前关节角度: {[f'{a:.2f}°' for a in current]}")
            else:
                self.get_logger().warn(f"容差 {label} 规划失败")

            time.sleep(1.5)

        self.move_to_target("home")

    def run_demo_7a(self):
        """演示 7a: 仅位置约束（三种容差对比）"""
        from moveit.core.kinematic_constraints import construct_link_constraint

        self.get_logger().info("=== 演示 7a: 仅位置约束（三种容差对比） ===")
        self.move_to_target("home")
        time.sleep(1.0)

        pos_tolerances = [0.001, 0.01, 0.05]
        pos_labels = ["严格(±1mm)", "中等(±10mm)", "宽松(±50mm)"]

        for tol, label in zip(pos_tolerances, pos_labels):
            self.get_logger().info(f"\n  7a-{label}: 目标 [0.25, 0.1, 0.3]m, 位置容差={tol}m")
            position_only = construct_link_constraint(
                link_name="link6", source_frame="base_link",
                cartesian_position=[0.25, 0.1, 0.3],
                cartesian_position_tolerance=tol,
            )
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(motion_plan_constraints=[position_only])
            plan_result = self.arm.plan()
            if plan_result:
                self.robot.execute(group_name=self.planning_group, robot_trajectory=plan_result.trajectory)
                time.sleep(0.5)
                pose = self.get_current_pose()
                self.get_logger().info(
                    f"  {label} 成功 → 实际位置: [{pose['position'][0]:.1f}, "
                    f"{pose['position'][1]:.1f}, {pose['position'][2]:.1f}]mm"
                )
            else:
                self.get_logger().warn(f"  {label} 规划失败")
            time.sleep(1.0)

        self.move_to_target("home")

    def run_demo_7b(self):
        """演示 7b: 仅姿态约束（三种容差对比）"""
        from moveit.core.kinematic_constraints import construct_link_constraint

        self.get_logger().info("=== 演示 7b: 仅姿态约束（三种容差对比） ===")
        self.move_to_target("home")
        time.sleep(1.0)

        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(
            np.pi, 0.0, np.pi / 2, axes='sxyz'
        )

        ori_tolerances = [0.05, 0.1, 0.3]
        ori_labels = ["严格(±2.9°)", "中等(±5.7°)", "宽松(±17.2°)"]

        for tol, label in zip(ori_tolerances, ori_labels):
            self.get_logger().info(f"\n  7b-{label}: 目标末端朝下, 姿态容差={tol}rad")
            orientation_only = construct_link_constraint(
                link_name="link6", source_frame="base_link",
                orientation=[qx, qy, qz, qw],
                orientation_tolerance=tol,
            )
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(motion_plan_constraints=[orientation_only])
            plan_result = self.arm.plan()
            if plan_result:
                self.robot.execute(group_name=self.planning_group, robot_trajectory=plan_result.trajectory)
                time.sleep(0.5)
                pose = self.get_current_pose()
                self.get_logger().info(
                    f"  {label} 成功 → 实际旋转: [{pose['rotation_deg'][0]:.1f}°, "
                    f"{pose['rotation_deg'][1]:.1f}°, {pose['rotation_deg'][2]:.1f}°]"
                )
            else:
                self.get_logger().warn(f"  {label} 规划失败")
            time.sleep(1.0)

        self.move_to_target("home")

    def run_demo_7c(self):
        """演示 7c: 位置+姿态约束（三种容差对比）"""
        from moveit.core.kinematic_constraints import construct_link_constraint

        self.get_logger().info("=== 演示 7c: 位置+姿态约束（三种容差对比） ===")
        self.move_to_target("home")
        time.sleep(1.0)

        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(
            np.pi, 0.0, np.pi / 2, axes='sxyz'
        )

        combo_tolerances = [(0.001, 0.05), (0.01, 0.1), (0.05, 0.3)]
        combo_labels = ["严格(±1mm,±2.9°)", "中等(±10mm,±5.7°)", "宽松(±50mm,±17.2°)"]

        for (pos_tol, ori_tol), label in zip(combo_tolerances, combo_labels):
            self.get_logger().info(
                f"\n  7c-{label}: 目标 [0.3,-0.1,0.25]m+朝下, "
                f"位置容差={pos_tol}m, 姿态容差={ori_tol}rad"
            )
            full_constraint = construct_link_constraint(
                link_name="link6", source_frame="base_link",
                cartesian_position=[0.3, -0.1, 0.25],
                cartesian_position_tolerance=pos_tol,
                orientation=[qx, qy, qz, qw],
                orientation_tolerance=ori_tol,
            )
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(motion_plan_constraints=[full_constraint])
            plan_result = self.arm.plan()
            if plan_result:
                self.robot.execute(group_name=self.planning_group, robot_trajectory=plan_result.trajectory)
                time.sleep(0.5)
                pose = self.get_current_pose()
                self.get_logger().info(
                    f"  {label} 成功 → 位置: [{pose['position'][0]:.1f}, "
                    f"{pose['position'][1]:.1f}, {pose['position'][2]:.1f}]mm, "
                    f"旋转: [{pose['rotation_deg'][0]:.1f}°, "
                    f"{pose['rotation_deg'][1]:.1f}°, {pose['rotation_deg'][2]:.1f}°]"
                )
            else:
                self.get_logger().warn(f"  {label} 规划失败")
            time.sleep(1.0)

        self.move_to_target("home")

    def run_demo_7(self):
        """演示 7: 链接约束目标（位置/姿态/位置+姿态）"""
        self.get_logger().info("=== 演示 7: 链接约束目标 ===")
        self.run_demo_7a()
        self.run_demo_7b()
        self.run_demo_7c()

    def _move_with_path_constraint(self, tilt, target_pos, from_pos, label=""):
        """先无约束调整当前位置姿态，再以路径约束运动到目标位置"""
        tilt_deg = np.rad2deg(tilt)
        rot = [np.pi, tilt, np.pi / 2]

        # 1) 先在当前位置无约束调整到目标姿态（确保起始满足路径约束）
        self.get_logger().info(f"{label}: 调整姿态到倾斜 {tilt_deg:.1f}°...")
        self.move_to_pose(from_pos[0], from_pos[1], from_pos[2],
                          rot[0], rot[1], rot[2])
        # time.sleep(0.5)

        # 2) 构建路径姿态约束
        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(
            np.pi, tilt, np.pi / 2, axes='sxyz'
        )
        path_constraints = Constraints()
        path_constraints.name = f"keep_tilt_{tilt_deg:.0f}"

        oc = OrientationConstraint()
        oc.header.frame_id = "base_link"
        oc.link_name = "link6"
        oc.orientation.x, oc.orientation.y = qx, qy
        oc.orientation.z, oc.orientation.w = qz, qw
        oc.absolute_x_axis_tolerance = 0.26   # ±15°
        oc.absolute_y_axis_tolerance = 0.26   # ±15°
        oc.absolute_z_axis_tolerance = 3.14   # 自由旋转
        oc.weight = 1.0
        path_constraints.orientation_constraints.append(oc)

        self.arm.set_path_constraints(path_constraints=path_constraints)

        # 3) 构建目标位姿（同样的倾斜角）
        qx_t, qy_t, qz_t, qw_t = tf_transformations.quaternion_from_euler(
            rot[0], rot[1], rot[2], axes='sxyz'
        )
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose.position.x = target_pos[0] / 1000.0
        pose_goal.pose.position.y = target_pos[1] / 1000.0
        pose_goal.pose.position.z = target_pos[2] / 1000.0
        pose_goal.pose.orientation.x = qx_t
        pose_goal.pose.orientation.y = qy_t
        pose_goal.pose.orientation.z = qz_t
        pose_goal.pose.orientation.w = qw_t

        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="link6")

        self.get_logger().info(f"{label}: 倾斜 {tilt_deg:.1f}° 带约束运动中...")
        plan_result = self.arm.plan()
        if plan_result:
            self.robot.execute(group_name=self.planning_group, robot_trajectory=plan_result.trajectory)
            self.get_logger().info(f"{label} 成功")
        else:
            self.get_logger().warn(f"{label} 规划失败")

        # 清除路径约束
        self.arm.set_path_constraints(path_constraints=Constraints())
        time.sleep(0.5)

    def run_demo_8(self):
        """演示 8: 路径约束 — 搬水杯场景（3种姿态交替搬运 A↔B）"""
        self.get_logger().info("=== 演示 8: 路径约束（3种姿态交替搬运） ===")

        pos_a = [250.0, 150.0, 250.0]
        pos_b = [250.0, -150.0, 250.0]

        # 3种倾斜姿态（差异明显）
        tilts = [0.0, 0.35, -0.35]   # 0°, ~20°, ~-20°
        labels = ["姿态1(竖直)", "姿态2(前倾20°)", "姿态3(后倾20°)"]

        # 姿态1: A → B（从A出发，先调整到姿态1再带约束走到B）
        self.get_logger().info(f"--- 第1段: {labels[0]} A→B ---")
        self._move_with_path_constraint(tilts[0], pos_b, pos_a, f"{labels[0]} A→B")

        # 姿态2: B → A（在B处调整到姿态2再带约束走到A）
        self.get_logger().info(f"--- 第2段: {labels[1]} B→A ---")
        self._move_with_path_constraint(tilts[1], pos_a, pos_b, f"{labels[1]} B→A")

        # 姿态3: A → B（在A处调整到姿态3再带约束走到B）
        self.get_logger().info(f"--- 第3段: {labels[2]} A→B ---")
        self._move_with_path_constraint(tilts[2], pos_b, pos_a, f"{labels[2]} A→B")

        self.move_to_target("home")

    # ==================== 演示调度 ====================

    def run_demo_sequence(self, demo_group: str = 'all'):
        """运行演示序列"""
        self.get_logger().info(f"\n========== 开始运行演示组: {demo_group} ==========\n")

        demos = {
            '1': self.run_demo_1, '2': self.run_demo_2, '3': self.run_demo_3,
            '4': self.run_demo_4, '5': self.run_demo_5, '6': self.run_demo_6,
            '7': self.run_demo_7, '7a': self.run_demo_7a, '7b': self.run_demo_7b,
            '7c': self.run_demo_7c, '8': self.run_demo_8,
        }

        if demo_group == 'all':
            for demo_func in demos.values():
                demo_func()
        elif demo_group in demos:
            demos[demo_group]()
        else:
            self.get_logger().error(
                f"未知演示组: {demo_group}. 可用: 'all', '1'-'8', '7a', '7b', '7c'"
            )
            return

        self.get_logger().info("\n演示序列完成")


def main(args=None):
    """入口函数"""
    rclpy.init(args=args)
    node = None
    try:
        node = MoveItAPIDemo()
        node.get_logger().info(f"运行演示组: {node.demo_group}")
        node.run_demo_sequence(node.demo_group)
        node.get_logger().info("演示完成，正在退出...")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"错误: {e}")
    finally:
        # MoveItPy C++ 析构存在已知 segfault 问题，
        # 直接强制退出进程，跳过所有 Python/C++ 析构
        os._exit(0)


if __name__ == '__main__':
    main()
