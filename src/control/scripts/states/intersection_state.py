#!/usr/bin/env python3

import rospy
import smach
import actionlib

from control.msg import ControlAction, ControlGoal
from actionlib_msgs.msg import GoalStatus

class IntersectionState(smach.State):
    def __init__(self, ac_client, topic_data, range_index):
        smach.State.__init__(
            self,
            outcomes = ['exit_intersection',
                        'preempted']
        )
        self._ac_client = ac_client
        self.topic_data = topic_data
        self.range_index = range_index

    def execute(self, userdata):
        # rospy.loginfo("[IntersectionState] Enter state: Intersection driving")

        goal = ControlGoal(mode="INTERSECTION")
        self._ac_client.send_goal(goal)
        # rospy.loginfo("[IntersectionState] Sent goal: Intersection mode. Now indefinite driving...")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():            
            if not self.topic_data['closest_index'] in self.range_index['intersection']:
                # rospy.loginfo("[IntersectionState] Intersection exit detected -> transitioning to urban_state")
                self._ac_client.cancel_goal()
                return 'exit_intersection'

            if self.preempt_requested():
                self.service_preempt()
                self._ac_client.cancel_goal()
                return 'preempted'

            state = self._ac_client.get_state()
            if state in [GoalStatus.ABORTED, GoalStatus.REJECTED]:
                # rospy.logwarn("[IntersectionState] Action ended unexpectedly (state=%s).", state)
                return 'preempted'
            elif state == GoalStatus.SUCCEEDED:
                # rospy.loginfo("[IntersectionState] Action ended with success (unexpected?).")
                return 'preempted'

            rate.sleep()

        self._ac_client.cancel_goal()
        return 'preempted'
