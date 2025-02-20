#!/usr/bin/env python3

import rospy
import smach
import actionlib

from control.msg import ControlAction, ControlGoal
from actionlib_msgs.msg import GoalStatus

class BuslaneState(smach.State):
    def __init__(self, ac_client, topic_data, range_index):
        smach.State.__init__(
            self,
            outcomes = ['exit_buslane',
                        'preempted']
        )
        self._ac_client = ac_client
        self.topic_data = topic_data
        self.range_index = range_index 

    def execute(self, userdata):
        # rospy.loginfo("[BuslaneState] Enter state: Buslane driving")

        goal = ControlGoal(mode="BUSLANE")
        self._ac_client.send_goal(goal)
        # rospy.loginfo("[BuslaneState] Sent goal: Buslane mode. Now indefinite driving...")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():            
            if self.topic_data['closest_index'] == self.range_index['buslane'][1]:
                # rospy.loginfo("[BuslaneState] Buslane exit detected -> transitioning to urban_state")
                self._ac_client.cancel_goal()
                return 'exit_buslane'

            if self.preempt_requested():
                self.service_preempt()
                self._ac_client.cancel_goal()
                return 'preempted'

            state = self._ac_client.get_state()
            if state in [GoalStatus.ABORTED, GoalStatus.REJECTED]:
                # rospy.logwarn("[BuslaneState] Action ended unexpectedly (state=%s).", state)
                return 'preempted'
            elif state == GoalStatus.SUCCEEDED:
                # rospy.loginfo("[BuslaneState] Action ended with success (unexpected?).")
                return 'preempted'

            rate.sleep()

        self._ac_client.cancel_goal()
        return 'preempted'
