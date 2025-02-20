#!/usr/bin/env python3

import rospy
import rospkg
import smach
import smach_ros
import actionlib

from actionlib_msgs.msg import GoalStatus
from control.msg import ControlAction

from states.urban_state import UrbanState
from states.crosswalk_state import CrosswalkState
from states.highway_state import HighwayState
from states.intersection_state import IntersectionState
from states.parking_state import ParkingState
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
        rospy.init_node('smach_action_client')

        rospack = rospkg.RosPack()
        package_path = rospack.get_path('control')

        default_graphml_path = os.path.join(package_path, 'config', 'Competition_track_graph.graphml')
        default_range_data = os.path.join(package_path, 'config', 'range_data.yaml')

        self.path = []
        self.current_pos = (0.0, 0.0)
        self.current_yaw = 0.0  # radian
        self.path_received = False
        self.gps_received = False

        self.topic_data = {'closest_index' : 0}
        self.graph = self.load_graphml_file(default_graphml_path)
        self.range_data = self.load_range_data_file(default_range_data)
        self.range_index = {}

        # ROS Subscribers
        self.path_sub = rospy.Subscriber('/global_path', Path, self.path_callback)
        self.gps_sub = rospy.Subscriber('/automobile/localisation', localisation, self.gps_callback)

        self.client = actionlib.SimpleActionClient('control_action', ControlAction)
        rospy.loginfo("Waiting for action server [control_action]...")
        self.client.wait_for_server()
        rospy.loginfo("Action server connected.")

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
                    # 'stop_state'            : 'STOP_STATE',
                    'exit_parking'          : 'URBAN_STATE',
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
        if not self.path_received:
            self.path = []
            for pose_stamped in msg.poses:
                x = pose_stamped.pose.position.x
                y = pose_stamped.pose.position.y
                self.path.append((x, y))
            
            # 각 segment별로 range_data 처리
            for segment, coords in self.range_data.items():
                # highway와 같이 시작, 끝 좌표가 튜플 형태로 저장된 경우
                if isinstance(coords, tuple):
                    start_coords, end_coords = coords
                    start_index = get_index(self.path, start_coords[0], start_coords[1])
                    end_index = get_index(self.path, end_coords[0], end_coords[1])
                    if None in (start_index, end_index):
                        self.range_index[segment] = (-1, -1)
                        continue
                    self.range_index[segment] = (start_index, end_index)
                    rospy.loginfo("Segment '%s' updated: start index=%s, end index=%s", 
                                segment, start_index, end_index)
                # 교차로나 노드 기반 구간의 경우, coords가 여러 노드의 좌표 리스트임
                elif isinstance(coords, list):
                    node_indices = []
                    for node in coords:
                        idx = get_index(self.path, node[0], node[1])
                        if not idx:
                            continue
                        node_indices.append(idx)
                    node_indices = list(set(node_indices))
                    self.range_index[segment] = node_indices
                    rospy.loginfo("Segment '%s' updated with node indices: %s", segment, node_indices)
                else:
                    rospy.logwarn("Segment '%s' has an unknown coordinate format", segment)
            
            self.path_received = True


    def gps_callback(self, msg):
        if self.path_received:
            self.current_pos = (msg.posA, msg.posB)
            self.topic_data['closest_index'] = get_nearest_index(self.path, self.current_pos[0], self.current_pos[1])
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
            rospy.logwarn(f"Range data file {file_path} not found. Using an empty list.")
            return {}
        try:
            with open(file_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            range_data = {}
            for segment, seg_data in yaml_data.items():
                # highway와 같이 시작/끝 노드 번호가 있는 경우
                if 'start_node' in seg_data and 'end_node' in seg_data:
                    start_node = str(seg_data['start_node'])
                    end_node = str(seg_data['end_node'])
                    if start_node in self.graph.nodes and end_node in self.graph.nodes:
                        # GraphML에서는 attr.name에 따라 'x'와 'y'로 저장됨
                        start_x = float(self.graph.nodes[start_node].get('x', None))
                        start_y = float(self.graph.nodes[start_node].get('y', None))
                        end_x   = float(self.graph.nodes[end_node].get('x', None))
                        end_y   = float(self.graph.nodes[end_node].get('y', None))
                        range_data[segment] = ((start_x, start_y), (end_x, end_y))
                        rospy.loginfo("Loaded segment '%s': start node %s at (%s, %s), end node %s at (%s, %s)", 
                                    segment, start_node, start_x, start_y, end_node, end_x, end_y)
                    else:
                        rospy.logwarn("Nodes %s or %s not found in graph", start_node, end_node)
                # 교차로 등 여러 노드 번호로 구성된 구간인 경우
                elif 'nodes' in seg_data:
                    node_ids = seg_data['nodes']
                    nodes_coords = []
                    for node_id in node_ids:
                        node_id = str(node_id)
                        if node_id in self.graph.nodes:
                            x = float(self.graph.nodes[node_id].get('x', None))
                            y = float(self.graph.nodes[node_id].get('y', None))
                            nodes_coords.append((x, y))
                        else:
                            rospy.logwarn("Node %s not found in graph", node_id)
                    range_data[segment] = nodes_coords
                    rospy.loginfo("Loaded segment '%s' with nodes: %s", segment, nodes_coords)
                else:
                    rospy.logwarn("Segment '%s' has no valid node specification", segment)
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

