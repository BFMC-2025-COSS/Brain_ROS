#!/usr/bin/env python3

import rospy
import actionlib

from control.msg import ControlAction, ControlResult, ControlFeedback

from nav_msgs.msg import Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker
# from geometry_msgs.msg import Point
from utils.msg import localisation

from pure_pursuit import PurePursuit

import math
import json
# import os

class ControlServer:
    def __init__(self, name='control_action'):
        # ROS Node
        rospy.init_node('control_action_server')

        # ROS Parameters
        self.look_ahead_dist = rospy.get_param('~look_ahead_dist', 0.068)
        self.wheel_base = rospy.get_param('~wheel_base', 0.034)
        self.desired_speed = rospy.get_param('~desired_speed', 20.0)

        # ROS Subscribers
        self.path_sub = rospy.Subscriber('/global_path', Path, self.path_callback)
        self.gps_sub = rospy.Subscriber('/automobile/localisation', localisation, self.gps_callback)

        # ROS Publishers
        self.command_pub = rospy.Publisher('/automobile/command', String, queue_size=1)
        self.current_pos_pub = rospy.Publisher('/visualization/current_pos', Marker, queue_size=1)
        self.look_ahead_pub = rospy.Publisher('/visualization/look_ahead', Marker, queue_size=1)
        self.path_marker_pub = rospy.Publisher('/visualization/look_ahead_line', Marker, queue_size=1)

        # Internal variables
        self.path = []  # global path
        self.current_pos = (0.0, 0.0)
        self.current_yaw = 0.0  # radian
        # self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)

        self.pp = PurePursuit(self.look_ahead_dist, self.wheel_base)

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

    def normalize_angle(self, angle):
        while angle >= math.pi:
            angle -= 2.0 * math.pi
        while angle <= math.pi:
            angle += 2.0 * math.pi
        return angle

    def path_callback(self, msg):
        self.path = []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            self.path.append((x, y))

    def gps_callback(self, msg):
        self.current_pos = (msg.posA, msg.posB)
        self.current_yaw = self.normalize_angle(msg.rotA)

    def control_loop(self, event):
        if not self.path:
            return

        steering_angle = self.pp.compute_steering_angle(
            self.path,
            self.current_pos,
            self.current_yaw
        )

        command = {}
        command['action'] =  '1'
        command['speed'] = float(self.desired_speed / 100.0)
        command = json.dumps(command)
        self.command_pub.publish(command)

        command = {}
        command['action'] = '2'
        command['steerAngle'] = float(-1.0 * math.degrees(steering_angle))
        command = json.dumps(command)
        self.command_pub.publish(command)

    def execute_cb(self, goal):
        rospy.loginfo("[ControlActionServer] Received goal: mode=%s", goal.mode)

        feedback = ControlFeedback()
        result = ControlResult()

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self._as.is_preempt_requested():
                rospy.logwarn("[ControlActionServer] Preempt requested -> Cancel current goal.")
                self._as.set_preempted()
                return

            if goal.mode == "STOP":
                self.desired_speed = 0.0
                steering_angle = 0.0
            elif goal.mode == "URBAN":
                self.desired_speed = 20.0
                steering_angle = self.pp.compute_steering_angle(
                    self.path,
                    self.current_pos,
                    self.current_yaw
                )
            elif goal.mode == "CROSSWALK":
                self.desired_speed = 20.0
                steering_angle = self.pp.compute_steering_angle(
                    self.path,
                    self.current_pos,
                    self.current_yaw
                )
            elif goal.mode == "HIGHWAY":
                self.desired_speed = 20.0
                steering_angle = self.pp.compute_steering_angle(
                    self.path,
                    self.current_pos,
                    self.current_yaw
                )
            elif goal.mode == "INTERSECTION":
                self.desired_speed = 20.0
                steering_angle = self.pp.compute_steering_angle(
                    self.path,
                    self.current_pos,
                    self.current_yaw
                )
            elif goal.mode == "PARKING":
                self.desired_speed = 20.0
                steering_angle = self.pp.compute_steering_angle(
                    self.path,
                    self.current_pos,
                    self.current_yaw
                )
            elif goal.mode == "ROUNDABOUT":
                self.desired_speed = 20.0
                steering_angle = self.pp.compute_steering_angle(
                    self.path,
                    self.current_pos,
                    self.current_yaw
                )
            elif goal.mode == "RAMP":
                self.desired_speed = 20.0
                steering_angle = self.pp.compute_steering_angle(
                    self.path,
                    self.current_pos,
                    self.current_yaw
                )
            elif goal.mode == "BUSLANE":
                self.desired_speed = 20.0
                steering_angle = self.pp.compute_steering_angle(
                    self.path,
                    self.current_pos,
                    self.current_yaw
                )
            elif goal.mode == "TUNNEL":
                self.desired_speed = 20.0
                steering_angle = self.pp.compute_steering_angle(
                    self.path,
                    self.current_pos,
                    self.current_yaw
                )
            
            command = {}
            command['action'] =  '1'
            command['speed'] = float(self.desired_speed / 100.0)
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
