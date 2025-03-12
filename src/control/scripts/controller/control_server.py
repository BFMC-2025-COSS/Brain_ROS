#!/usr/bin/env python3

import rospy
import actionlib

from control.msg import ControlAction, ControlResult, ControlFeedback

from nav_msgs.msg import Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker
# from geometry_msgs.msg import Point
from utils.msg import localisation
from sensor_msgs.msg import Imu

from pure_pursuit import PurePursuit
from mpc import NonlinearMPCController

import math
import json
# import os

class ControlServer:
    def __init__(self, name='control_action'):
        # ROS Node
        rospy.init_node('control_action_server')

        # ROS Parameters
        self.look_ahead_dist = rospy.get_param('~look_ahead_dist', 0.38)
        self.wheel_base = rospy.get_param('~wheel_base', 0.26)
        self.desired_speed = rospy.get_param('~desired_speed', 0.3)

        # ROS Subscribers
        self.path_sub = rospy.Subscriber('/global_path', Path, self.path_callback)
        self.gps_sub = rospy.Subscriber('/automobile/localisation', localisation, self.gps_callback)
        self.imu_sub = rospy.Subscriber("/automobile/IMU", Imu, self.imu_callback, queue_size=1)

        # ROS Publishers
        self.command_pub = rospy.Publisher('/automobile/command', String, queue_size=10)
        self.current_pos_pub = rospy.Publisher('/visualization/current_pos', Marker, queue_size=1)
        self.look_ahead_pub = rospy.Publisher('/visualization/look_ahead', Marker, queue_size=1)
        self.path_marker_pub = rospy.Publisher('/visualization/look_ahead_line', Marker, queue_size=1)

        # Internal variables
        self.path_received = False
        self.gps_received = False
        self.imu_received = False

        self.path = []  # global path
        self.current_pos = (0.0, 0.0)
        self.current_yaw = 0.0  # radian

        self.pp = PurePursuit(self.look_ahead_dist, self.wheel_base)
        self.mpc = NonlinearMPCController(dt=0.25, horizon=10, wheelbase=0.26)

        ## Parking
        graph_nodes = {
            "231": (10.82, 0.92),
            "900": (9.34, 0.54),
            "910": (9.34, 1.3),
            # 글로벌 경로 등 필요한 노드 추가
        }
        graph_edges = {
            "231": ["900", "910"],
            "900": ["231"],
            "910": ["231"]
        }
        self.parking_path = []
        self.parking_path = self.mpc.build_parking_path("231", "900", graph_nodes, graph_edges)

        # Action
        self._action_name = name
        self._as = actionlib.SimpleActionServer(
            self._action_name,
            ControlAction,
            execute_cb=self.execute_cb,
            auto_start=False
        )
        self._as.start()

        rospy.loginfo("[ControlActionServer] '%s' action server started.", self._action_name)

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)

        return math.atan2(siny_cosp, cosy_cosp)

    def path_callback(self, msg):
        if self.path_received:
            return
        
        self.path = []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            self.path.append((x, y))

        self.path_received = True

    def gps_callback(self, msg):
        if not self.imu_received:
            return
        
        x_center, y_center = msg.posA, msg.posB

        x_rear = x_center - (self.wheel_base / 2) * math.cos(self.current_yaw)
        y_rear = y_center - (self.wheel_base / 2) * math.sin(self.current_yaw)

        self.current_pos = (x_rear, y_rear)

        self.gps_received = True

    def imu_callback(self, msg: Imu):
        q = msg.orientation
        self.current_yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

        self.imu_received = True

    def execute_cb(self, goal):
        rospy.loginfo("[ControlActionServer] Received goal: mode=%s", goal.mode)

        feedback = ControlFeedback()
        result = ControlResult()

        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self._as.is_preempt_requested():
                rospy.logwarn("[ControlActionServer] Preempt requested -> Cancel current goal.")
                self._as.set_preempted()
                return

            if goal.mode == "STOP":
                speed = 0.0
                steering_angle = 0.0
            elif goal.mode == "URBAN":
                speed, steering_angle = self.mpc.compute_control_command(
                    self.path,
                    self.current_pos,
                    self.current_yaw,
                    self.desired_speed
                )
            elif goal.mode == "CROSSWALK":
                speed, steering_angle = self.mpc.compute_control_command(
                    self.path,
                    self.current_pos,
                    self.current_yaw,
                    self.desired_speed
                )
            elif goal.mode == "HIGHWAY":
                speed, steering_angle = self.mpc.compute_control_command(
                    self.path,
                    self.current_pos,
                    self.current_yaw,
                    self.desired_speed
                )
            elif goal.mode == "INTERSECTION":
                speed, steering_angle = self.mpc.compute_control_command(
                    self.path,
                    self.current_pos,
                    self.current_yaw,
                    self.desired_speed
                )
            elif goal.mode == "PARKING":
                speed, steering_angle = self.mpc.compute_control_command(
                    self.parking_path,
                    self.current_pos,
                    self.current_yaw,
                    desired_speed=0.2,
                    scenario="parking"
                )

                if len(self.parking_path) > 0:
                    last_x, last_y = self.parking_path[-1]
                    dist_to_end = math.hypot(self.current_pos[0] - last_x,
                                             self.current_pos[1] - last_y)
                    yaw_error = abs(self.current_yaw)

                    if dist_to_end < 0.15 and yaw_error < math.radians(10):
                        self.exit_parking_path = list(reversed(self.parking_path))

                        feedback.status_message = "Parking complete!"
                        self._as.publish_feedback(feedback)

                        result.success = True
                        rospy.loginfo("[ControlActionServer] Parking done. (dist=%.2f)", dist_to_end)
                        self._as.set_succeeded(result)
                        return
                    
            elif goal.mode == "EXIT_PARKING":
                speed, steering_angle = self.mpc.compute_control_command(
                    self.exit_parking_path,
                    self.current_pos,
                    self.current_yaw,
                    desired_speed=0.2,
                    scenario="exit_parking"
                )

                if len(self.exit_parking_path) > 0:
                    last_x, last_y = self.exit_parking_path[-1]
                    dist_to_end = math.hypot(self.current_pos[0] - last_x,
                                             self.current_pos[1] - last_y)

                    if dist_to_end < 0.15:
                        feedback.status_message = "Exit Parking complete!"
                        self._as.publish_feedback(feedback)

                        result.success = True
                        rospy.loginfo("[ControlActionServer] Exit Parking done. (dist=%.2f)", dist_to_end)
                        self._as.set_succeeded(result)
                        return

            elif goal.mode == "ROUNDABOUT":
                speed, steering_angle = self.mpc.compute_control_command(
                    self.path,
                    self.current_pos,
                    self.current_yaw,
                    self.desired_speed
                )
            elif goal.mode == "RAMP":
                speed, steering_angle = self.mpc.compute_control_command(
                    self.path,
                    self.current_pos,
                    self.current_yaw,
                    self.desired_speed
                )
            elif goal.mode == "BUSLANE":
                speed, steering_angle = self.mpc.compute_control_command(
                    self.path,
                    self.current_pos,
                    self.current_yaw,
                    self.desired_speed
                )
            elif goal.mode == "TUNNEL":
                speed, steering_angle = self.mpc.compute_control_command(
                    self.path,
                    self.current_pos,
                    self.current_yaw,
                    self.desired_speed
                )
            
            command = {}
            command['action'] =  '1'
            command['speed'] = float(speed)
            command = json.dumps(command)
            self.command_pub.publish(command)

            command = {}
            command['action'] = '2'
            command['steerAngle'] = float(-1.0 * math.degrees(steering_angle))
            command = json.dumps(command)
            self.command_pub.publish(command)

            feedback.status_message = "Current mode: {}".format(goal.mode)
            self._as.publish_feedback(feedback)

            rate.sleep()

        result.success = True
        self._as.set_succeeded(result)
        rospy.loginfo("[ControlActionServer] Node is shutting down. Goal ended with success.")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        cs = ControlServer()
        cs.run()
    except rospy.ROSInterruptException:
        pass
