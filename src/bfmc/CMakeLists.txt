cmake_minimum_required(VERSION 3.0.2)
project(bfmc)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  nav_msgs
  tf
  sensor_msgs
  std_msgs
  message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  realsense_imu.msg
  bfmc_imu.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
  geometry_msgs
)



## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
catkin_install_python(PROGRAMS
  Brain/src/hardware/camera/realsense_publisher.py
  Brain/src/hardware/YOLO/yolo_detector.py
  Brain/src/hardware/Lanekeep/lane_keeper.py
  Brain/src/hardware/serialhandler/serial_write.py
  Brain/src/hardware/serialhandler/serial_read.py
  Brain/src/hardware/serialhandler/command.py
  Brain/src/hardware/Localization/initial_odo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs geometry_msgs rospy
)