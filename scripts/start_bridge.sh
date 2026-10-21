#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
exec ros2 run ros_gz_bridge parameter_bridge \
  '/model/vehicle_blue/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist' \
  '/patrol/camera@sensor_msgs/msg/Image[ignition.msgs.Image' \
  '/model/vehicle_blue/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry'
