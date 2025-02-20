#!/usr/bin/env python3

import rospy
import smach
import actionlib

from control.msg import ControlAction, ControlGoal
from actionlib_msgs.msg import GoalStatus

class ParkingState(smach.State):
    def __init__(self, ac_client, topic_data, range_index):
        smach.State.__init__(
            self,
            outcomes = ['exit_parking',
                        'preempted']
        )
        self._ac_client = ac_client
        self.topic_data = topic_data
        self.range_index = range_index 

    def execute(self, userdata):
        # rospy.loginfo("[ParkingState] Enter state: Parking driving")

        goal = ControlGoal(mode="PARKING")
        self._ac_client.send_goal(goal)
        # rospy.loginfo("[ParkingState] Sent goal: Parking mode. Now indefinite driving...")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():            
            if self.topic_data['closest_index'] == self.range_index['parking'][1]:
                # rospy.loginfo("[ParkingState] Parking exit detected -> transitioning to urban_state")
                self._ac_client.cancel_goal()
                return 'exit_parking'

            if self.preempt_requested():
                self.service_preempt()
                self._ac_client.cancel_goal()
                return 'preempted'

            state = self._ac_client.get_state()
            if state in [GoalStatus.ABORTED, GoalStatus.REJECTED]:
                # rospy.logwarn("[ParkingState] Action ended unexpectedly (state=%s).", state)
                return 'preempted'
            elif state == GoalStatus.SUCCEEDED:
                # rospy.loginfo("[ParkingState] Action ended with success (unexpected?).")
                return 'preempted'

            rate.sleep()

        self._ac_client.cancel_goal()
        return 'preempted'
