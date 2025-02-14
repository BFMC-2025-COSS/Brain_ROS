#!/usr/bin/env python3

import rospy
import smach
import smach_ros
import actionlib

from actionlib_msgs.msg import GoalStatus
from control.msg import ControlAction
from states.urban_state import UrbanState

from nav_msgs.msg import Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker
# from geometry_msgs.msg import Point
from utils.msg import localisation

class TechnicalChallengeSmach:
    def __init__(self):
        rospy.init_node('smach_action_client_node')

        self.client = actionlib.SimpleActionClient('control_action', ControlAction)
        rospy.loginfo("Waiting for action server [control_action]...")
        self.client.wait_for_server()
        rospy.loginfo("Action server connected.")

        # ROS Subscribers
        self.path_sub = rospy.Subscriber('/global_path', Path, self.path_callback)
        self.gps_sub = rospy.Subscriber('/automobile/localisation', localisation, self.gps_callback)

        self.path = []
        self.current_pos = (0.0, 0.0)
        self.current_yaw = 0.0  # radian

        self.topic_data = {'closest_index' : 0}
        self.range_data = {'highway' : (0, 0)}

        self.sm_top = smach.StateMachine(
            outcomes=['SM_FINISHED', 'SM_PREEMPTED', 'HIGHWAY_STATE']
        )

        with self.sm_top:
            smach.StateMachine.add(
                'URBAN_STATE',
                UrbanState(ac_client=self.client, topic_data=self.topic_data, range_data=self.range_data),
                transitions={
                    'highway_state': 'HIGHWAY_STATE',
                    'preempted': 'SM_PREEMPTED'
                }
            )

        self.sis = smach_ros.IntrospectionServer('smach_viewer', self.sm_top, '/SM_TOP')
        self.sis.start()

    def path_callback(self, msg):
        self.path = []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            self.path.append((x, y))

        self.range_data['highway'] = (self.get_nearest_index(self.path, 13.94, 10.76), self.get_nearest_index(self.path, 6.7, 12.16))

    def gps_callback(self, msg):
        self.current_pos = (msg.posA, msg.posB)
        self.topic_data['closest_index'] = self.get_nearest_index(self.path, self.current_pos[0], self.current_pos[1])
        print(self.topic_data, self.range_data)

    def get_nearest_index(self, path, x, y):
        # Find the nearest path index to (x, y)
        if not path:
            return None
        dists = [(x - px)**2 + (y - py)**2 for (px, py) in path]
        min_dist = min(dists)
        return dists.index(min_dist)

    def run(self):
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

