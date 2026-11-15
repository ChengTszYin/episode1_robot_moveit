cmake_minimum_required(VERSION 3.8)
project(episode1_urdf_with_gripper)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  urdf
  meshes
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()