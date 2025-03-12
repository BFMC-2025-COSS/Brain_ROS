#!/usr/bin/env python3

import rospy
import smach
import actionlib

from control.msg import ControlAction, ControlGoal
from actionlib_msgs.msg import GoalStatus

class UrbanState(smach.State):
    def __init__(self, ac_client, topic_data, range_index):
        smach.State.__init__(
            self,
            outcomes = ['enter_crosswalk',
                        'enter_highway',
                        'enter_intersection',
                        'enter_parking',
                        'enter_roundabout',
                        'enter_buslane',
                        'preempted'
                        ]
        )
        self._ac_client = ac_client
        self.topic_data = topic_data
        self.range_index = range_index

    def execute(self, userdata):
        # rospy.loginfo("[UrbanState] Enter state: URBAN driving")

        # 1) 액션 Goal 전송 (무기한 주행)
        goal = ControlGoal(mode="URBAN")
        self._ac_client.send_goal(goal)
        # rospy.loginfo("[UrbanState] Sent goal: URBAN mode. Now indefinite driving...")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.topic_data['closest_index'] in self.range_index['crosswalk']:
                # rospy.loginfo("[UrbanState] Crosswalk entrance detected -> transition")
                self._ac_client.cancel_goal()
                return 'enter_crosswalk'
            
            if self.range_index['highway_1'][0] <= self.topic_data['closest_index'] < self.range_index['highway_1'][1] or \
                self.range_index['highway_2'][0] <= self.topic_data['closest_index'] < self.range_index['highway_2'][1]:
                # rospy.loginfo("[UrbanState] Highway entrance detected -> transition")
                self._ac_client.cancel_goal()
                return 'enter_highway'
            
            if self.topic_data['closest_index'] in self.range_index['intersection']:
                # rospy.loginfo("[UrbanState] Intersection entrance detected -> transition")
                self._ac_client.cancel_goal()
                return 'enter_intersection'
            
            if self.topic_data['closest_index'] == self.range_index['parking'][0]:
                # rospy.loginfo("[UrbanState] Parking entrance detected -> transition")
                self._ac_client.cancel_goal()
                return 'enter_parking'
            
            if self.topic_data['closest_index'] in self.range_index['roundabout']:
                # rospy.loginfo("[UrbanState] Roundabout entrance detected -> transition")
                self._ac_client.cancel_goal()
                return 'enter_roundabout'
            
            if self.range_index['buslane'][0] <= self.topic_data['closest_index'] < self.range_index['buslane'][1]:
                # rospy.loginfo("[UrbanState] Buslane entrance detected -> transition")
                self._ac_client.cancel_goal()
                return 'enter_buslane'

            if self.preempt_requested():
                self.service_preempt()
                self._ac_client.cancel_goal()
                return 'preempted'

            state = self._ac_client.get_state()
            if state in [GoalStatus.ABORTED, GoalStatus.REJECTED]:
                # rospy.logwarn("[UrbanState] Action ended unexpectedly (state=%s).", state)
                return 'preempted'
            elif state == GoalStatus.SUCCEEDED:
                # rospy.loginfo("[UrbanState] Action ended with success (unexpected?).")
                return 'preempted'

            rate.sleep()

        self._ac_client.cancel_goal()
        return 'preempted'
