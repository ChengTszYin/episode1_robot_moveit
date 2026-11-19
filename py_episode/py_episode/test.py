import os 
import yaml 
import numpy as np
from scipy.spatial.transform import Rotation

file_path = "/home/dllm/arms_ws/src/py_episode/config/T_camera2end.yaml"
with open(file_path, 'r')  as f:
    data = yaml.safe_load(f)
T_camera2end = np.array(data['T_camera2end'])
# print(T_camera2end)

position = [-1.2302910909056664, 31.437721103429794, 277.0000100135803]
orientation = [-0.5870468145346133, 0.6076074216152906, -0.3199483889506531, -0.4287450141408804]
rotate = Rotation.from_quat(orientation).as_matrix()
T_grasp2camera = np.eye(4)
T_grasp2camera[:3, :3] = rotate
T_grasp2camera[:3, 3] = position

T_grasp2end = T_grasp2camera @ T_camera2end
print(T_grasp2end) 

# def _task_entry_cb(self, request, response):
#         if self.robot_busy:
#             response.success = False
#             response.message = "Robot busy"
#             return response

#         if not request.poses:
#             response.success = False
#             response.message = "No poses in request"
#             return response

#         self.robot_busy = True
#         try:
#             for i, pose in enumerate(request.poses):
#                 # ----- build PoseStamped in camera_link -----
#                 t = [pose.position.x * 1000.0, pose.position.y * 1000.0, pose.position.z * 1000.0]
#                 q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
#                 # ----- following the tf process of 3.demo_VLM_handler.py -----
#                 T_grasp2camera = np.eye(4)
#                 T_grasp2camera[:3, :3] = Rotation.from_quat(q).as_matrix()
#                 T_grasp2camera[:, [0, 2]] = T_grasp2camera[:, [2, 0]]
#                 T_grasp2camera[:, 0] = -T_grasp2camera[:, 0]
#                 T_grasp2camera[:3, 3] = t
#                 T_grasp2end = self.T_camera2end @ T_grasp2camera
#                 
#                 transform = self.tf_buffer.lookup_transform(
#                     target_frame='base_link',          # or 'base'
#                     source_frame='gripper_base_link',              # or 'end_effector', 'gripper_link', etc. be careful
#                     time=Time(),                       # latest available
#                     timeout=rclpy.duration.Duration(seconds=1.0)
#                 )
#                 
#                 t_end2base = transform.transform.translation
#                 q_end2base = transform.transform.rotation
#                 R_end2base = Rotation.from_quat([q_end2base.x, q_end2base.y, q_end2base.z, q_end2base.w]).as_matrix()
#                 
#                 T_end2base = np.eye(4)
#                 T_end2base[:3, :3] = R_end2base
#                 T_end2base[:3, 3] = [t_end2base.x, t_end2base.y, t_end2base.z]
#                 T_grasp2base = T_end2base @ T_grasp2end
#                 
#                 # 构造舵机夹爪到末端的变换矩阵
#                 theta = 0
#                 alpha = 0
#                 d = 120
#                 a = 0
#                 T_servo2end = np.array(
#                     [
#                         [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
#                         [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
#                         [0, np.sin(alpha), np.cos(alpha), d],
#                         [0, 0, 0, 1]
#                     ]
#                 )
#                 T_end2base_new = T_grasp2base @ np.linalg.inv(T_servo2end)
#                 P_base = T_end2base_new[:3, 3]
#                 rotation = Rotation.from_matrix(T_end2base_new[:3, :3])
#                 euler = rotation.as_euler('XZY', degrees=True)
#                 x, y, z = P_base
#                 x_rot = np.deg2rad(euler[0])   # or whichever index is X in XZY
#                 y_rot = np.deg2rad(euler[1])
#                 z_rot = np.deg2rad(euler[2])
#                 # ----- the end of the tf process of 3.demo_VLM_handler.py -----
#                 if self.move_to_pose(x, y, z, 0.0, 0.0, 90.0):
#                     response.success = True
#                     response.message = f"Executed grasp {i+1}"
#                     return response

#             response.success = False
#             response.message = "All grasp poses failed to plan/execute"
#         except Exception as e:
#             self.get_logger().error(f"Service error: {e}")
#             response.success = False
#             response.message = f"error: {e}"
#         finally:
#             self.robot_busy = False
#         return response
