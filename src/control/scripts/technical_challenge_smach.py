#!/usr/bin/env python3

import rospy
import rospkg
import smach
import smach_ros
import actionlib

from actionlib_msgs.msg import GoalStatus
from control.msg import ControlAction

from states.stop_state import StopState
from states.urban_state import UrbanState
from states.crosswalk_state import CrosswalkState
from states.highway_state import HighwayState
from states.intersection_state import IntersectionState
from states.parking_state import ParkingState
from states.exit_parking_state import ExitParkingState
from states.roundabout_state import RoundaboutState
from states.buslane_state import BuslaneState

from nav_msgs.msg import Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker
# from geometry_msgs.msg import Point
from utils.msg import localisation

from path.path_functions import *
from control_utils.load_file import LoadData

import networkx as nx
import os
import yaml

class TechnicalChallengeSmach:
    def __init__(self):
        # ROS Initialization
        rospy.init_node('smach_action_client')

        rospack = rospkg.RosPack()
        package_path = rospack.get_path('control')

        default_graphml_path = os.path.join(package_path, 'config', 'Competition_track_graph.graphml')
        default_range_data = os.path.join(package_path, 'config', 'range_data.yaml')

        # ROS Parameters
        self.crosswalk_dist = rospy.get_param('~crosswalk_dist', 0.3)
        self.intersection_dist = rospy.get_param('~intersection_dist', 0.5)
        self.roundabout_dist = rospy.get_param('~roundabout_dist', 0.85)
        self.default_dist = rospy.get_param('~default_dist', 0.3)

        # Variables
        self.path = []
        self.current_pos = (0.0, 0.0)
        self.current_yaw = 0.0  # radian
        self.path_received = False
        self.gps_received = False

        self.ld = LoadData()
        self.topic_data = {'closest_index' : 0}
        self.graph = self.ld.load_graphml_file(default_graphml_path)
        self.range_data = self.ld.load_range_data_file(default_range_data, self.graph)
        self.range_index = {}

        self.segment_dist_map = {
            "crosswalk": self.crosswalk_dist,
            "intersection": self.intersection_dist,
            "roundabout": self.roundabout_dist
        }

        # ROS Subscribers
        self.path_sub = rospy.Subscriber('/global_path', Path, self.path_callback)
        self.gps_sub = rospy.Subscriber('/automobile/localisation', localisation, self.gps_callback)

        # ROS Action
        self.client = actionlib.SimpleActionClient('control_action', ControlAction)
        rospy.loginfo("Waiting for action server [control_action]...")
        self.client.wait_for_server()
        rospy.loginfo("Action server connected.")

        # SMACH
        self.sm_top = smach.StateMachine(
            outcomes=['SM_FINISHED', 'SM_PREEMPTED']
        )

        with self.sm_top:
            smach.StateMachine.add(
                'URBAN_STATE',
                UrbanState(ac_client=self.client, topic_data=self.topic_data, range_index=self.range_index),
                transitions={
                    # 'enter_stop'            : 'STOP_STATE',
                    'enter_crosswalk'       : 'CROSSWALK_STATE',
                    'enter_highway'         : 'HIGHWAY_STATE',
                    'enter_intersection'    : 'INTERSECTION_STATE',
                    'enter_parking'         : 'PARKING_STATE',
                    'enter_roundabout'      : 'ROUNDABOUT_STATE',
                    # 'enter_ramp'            : 'RAMP_STATE',
                    'enter_buslane'         : 'BUSLANE_STATE',
                    # 'enter_tunnel'          : 'TUNNEL_STATE',
                    'preempted'             : 'SM_PREEMPTED'
                }
            )

            smach.StateMachine.add(
                'CROSSWALK_STATE',
                CrosswalkState(ac_client=self.client, topic_data=self.topic_data, range_index=self.range_index),
                transitions={
                    # 'stop_state'            : 'STOP_STATE',
                    'exit_crosswalk'        : 'URBAN_STATE',
                    'preempted'             : 'SM_PREEMPTED'
                }
            )

            smach.StateMachine.add(
                'HIGHWAY_STATE',
                HighwayState(ac_client=self.client, topic_data=self.topic_data, range_index=self.range_index),
                transitions={
                    # 'stop_state'            : 'STOP_STATE',
                    'exit_highway'          : 'URBAN_STATE',
                    'preempted'             : 'SM_PREEMPTED'
                }
            )

            smach.StateMachine.add(
                'INTERSECTION_STATE',
                IntersectionState(ac_client=self.client, topic_data=self.topic_data, range_index=self.range_index),
                transitions={
                    # 'stop_state'            : 'STOP_STATE',
                    'exit_intersection'     : 'URBAN_STATE',
                    'preempted'             : 'SM_PREEMPTED'
                }
            )

            smach.StateMachine.add(
                'PARKING_STATE',
                ParkingState(ac_client=self.client, topic_data=self.topic_data, range_index=self.range_index),
                transitions={
                    'stop_after_park'       : 'PARKING_STOP_STATE',
                    'exit_parking'          : 'EXIT_PARKING_STATE',
                    'preempted'             : 'SM_PREEMPTED'
                }
            )

            smach.StateMachine.add(
                'PARKING_STOP_STATE',
                StopState(ac_client=self.client, stop_time=2.0),
                transitions={
                    'done'                  : 'EXIT_PARKING_STATE',
                    'preempted'             : 'SM_PREEMPTED'
                }
            )

            smach.StateMachine.add(
                'EXIT_PARKING_STATE',
                ExitParkingState(ac_client=self.client, topic_data=self.topic_data, range_index=self.range_index),
                transitions={
                    'return_to_urban_state' : 'URBAN_STATE',
                    'preempted'             : 'SM_PREEMPTED'
                }
            )

            smach.StateMachine.add(
                'ROUNDABOUT_STATE',
                RoundaboutState(ac_client=self.client, topic_data=self.topic_data, range_index=self.range_index),
                transitions={
                    # 'stop_state'            : 'STOP_STATE',
                    'exit_roundabout'           : 'URBAN_STATE',
                    'preempted'             : 'SM_PREEMPTED'
                }
            )

            smach.StateMachine.add(
                'BUSLANE_STATE',
                BuslaneState(ac_client=self.client, topic_data=self.topic_data, range_index=self.range_index),
                transitions={
                    # 'stop_state'            : 'STOP_STATE',
                    'exit_buslane'          : 'URBAN_STATE',
                    'preempted'             : 'SM_PREEMPTED'
                }
            )

        self.sis = smach_ros.IntrospectionServer('smach_viewer', self.sm_top, '/SM_TOP')
        self.sis.start()

    def path_callback(self, msg):
        if self.path_received:
            return

        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.range_index = self.ld.load_path_file(self.range_data, self.path, self.segment_dist_map, self.default_dist)

        self.path_received = True

    def gps_callback(self, msg):
        if not self.path_received:
            return
        self.current_pos = (msg.posA, msg.posB)
        idx = get_nearest_index(self.path, self.current_pos[0], self.current_pos[1])
        if idx is not None:
            self.topic_data['closest_index'] = idx
        else:
            rospy.logwarn("No nearest index found for current GPS position!")
        self.gps_received = True

    def run(self):
        rospy.loginfo("Waiting for initial global path and GPS data...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.path_received and self.gps_received:
                rospy.loginfo("Initial data received. Starting SMACH state machine.")
                break
            rate.sleep()

        outcome = self.sm_top.execute()
        rospy.loginfo("[SMACH] Finished with outcome: %s", outcome)

        self.sis.stop()
        rospy.spin()


if __name__ == '__main__':
    try:
        tcs = TechnicalChallengeSmach()
        tcs.run()
    except rospy.ROSInterruptException:
        pass

