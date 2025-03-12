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

        self.topic_data = {'closest_index' : 0}
        self.graph = self.load_graphml_file(default_graphml_path)
        self.range_data = self.load_range_data_file(default_range_data)
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

        for segment, seg_info in self.range_data.items():
            segment_type = seg_info.get('type', 'midpoint')
            coords_list = seg_info.get('coords', [])

            if segment_type == 'range':
                if len(coords_list) != 2:
                    self.range_index[segment] = (-1, -1)
                    continue

                start_coords = coords_list[0]
                end_coords   = coords_list[1]

                start_indices = get_indices_within_distance(self.path, start_coords[0], start_coords[1])
                end_indices   = get_indices_within_distance(self.path, end_coords[0], end_coords[1])

                if not start_indices or not end_indices:
                    self.range_index[segment] = (-1, -1)
                    rospy.logwarn("Segment '%s': Failed to find indices for start or end coordinates.", segment)
                    continue

                start_index = min(start_indices, key=lambda idx:
                                (self.path[idx][0] - start_coords[0])**2 + (self.path[idx][1] - start_coords[1])**2)
                end_index = min(end_indices, key=lambda idx:
                                (self.path[idx][0] - end_coords[0])**2 + (self.path[idx][1] - end_coords[1])**2)

                self.range_index[segment] = (start_index, end_index)
                rospy.loginfo("Segment '%s': [range] start index=%d, end index=%d", segment, start_index, end_index)

            elif segment_type == 'midpoint':
                dist = self.segment_dist_map.get(segment, self.default_dist)
                node_indices = []

                for (nx, ny) in coords_list:
                    near_inds = get_indices_within_distance(self.path, nx, ny, dist)
                    node_indices.extend(near_inds)

                node_indices = sorted(set(node_indices))
                self.range_index[segment] = node_indices
                rospy.loginfo("Segment '%s': [midpoint] found indices=%s (distance=%.2f)", segment, node_indices, dist)

            else:
                dist = self.segment_dist_map.get(segment, self.default_dist)
                node_indices = []

                for (nx, ny) in coords_list:
                    near_inds = get_indices_within_distance(self.path, nx, ny, dist)
                    node_indices.extend(near_inds)

                node_indices = sorted(set(node_indices))
                self.range_index[segment] = node_indices
                rospy.loginfo("Segment '%s': [default] found indices=%s (distance=%.2f)",
                            segment, node_indices, dist)

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

    def load_graphml_file(self, file_path):
        try:
            g = nx.read_graphml(file_path)
            rospy.loginfo("Successfully loaded GraphML.")
            return g
        except Exception as e:
            rospy.logerr(f"Failed to load GraphML file: {e}")
            return None
    
    def load_range_data_file(self, file_path):
        if not os.path.exists(file_path):
            rospy.logwarn(f"Range data file {file_path} not found. Using empty data.")
            return {}

        try:
            with open(file_path, 'r') as f:
                yaml_data = yaml.safe_load(f)

            range_data = {}

            for segment, seg_data in yaml_data.items():
                segment_type = seg_data.get('type', 'midpoint')

                if 'nodes' not in seg_data:
                    rospy.logwarn(f"Segment '{segment}' has no 'nodes' key. Skipping.")
                    continue
                nodes_list = seg_data['nodes']

                node_coords = []

                for sublist in nodes_list:
                    if not isinstance(sublist, list):
                        continue

                    if len(sublist) == 1:
                        node_id = str(sublist[0])
                        if node_id not in self.graph.nodes:
                            rospy.logwarn("Segment '%s': node_id '%s' not in graph.", segment, node_id)
                            continue

                        x = float(self.graph.nodes[node_id].get('x', 0.0))
                        y = float(self.graph.nodes[node_id].get('y', 0.0))
                        node_coords.append((x, y))
                        rospy.loginfo("Segment '%s': single node => %s => (%.3f, %.3f)", segment, node_id, x, y)

                    elif len(sublist) == 2:
                        node_id1 = str(sublist[0])
                        node_id2 = str(sublist[1])

                        if (node_id1 not in self.graph.nodes) or (node_id2 not in self.graph.nodes):
                            rospy.logwarn("Segment '%s': either '%s' or '%s' not in graph.", segment, node_id1, node_id2)
                            continue

                        x1 = float(self.graph.nodes[node_id1].get('x', 0.0))
                        y1 = float(self.graph.nodes[node_id1].get('y', 0.0))
                        x2 = float(self.graph.nodes[node_id2].get('x', 0.0))
                        y2 = float(self.graph.nodes[node_id2].get('y', 0.0))

                        if segment_type == "midpoint":
                            mid_x = (x1 + x2) / 2.0
                            mid_y = (y1 + y2) / 2.0
                            node_coords.append((mid_x, mid_y))
                            rospy.loginfo("Segment '%s': 2-node midpoint => (%.3f, %.3f)", segment, mid_x, mid_y)
                        elif segment_type == "range":
                            node_coords.append((x1, y1))
                            node_coords.append((x2, y2))
                            rospy.loginfo("Segment '%s': 2-node range => start=(%.3f, %.3f), end=(%.3f, %.3f)", segment, x1, y1, x2, y2)
                        else:
                            rospy.logwarn("Segment '%s': unknown type '%s'", segment, segment_type)

                range_data[segment] = {
                    "type": segment_type,
                    "coords": node_coords
                }
                rospy.loginfo("Segment '%s' => node_coords list = %s", segment, node_coords)

            return range_data

        except Exception as e:
            rospy.logerr(f"Failed to load Range data file: {e}")
            return {}

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

