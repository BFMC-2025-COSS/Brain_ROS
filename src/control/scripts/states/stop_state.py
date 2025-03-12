#!/usr/bin/env python3

import rospy
import smach
import actionlib

from control.msg import ControlAction, ControlGoal
from actionlib_msgs.msg import GoalStatus

class StopState(smach.State):
    def __init__(self, ac_client, stop_time=None, condition_func=None):
        smach.State.__init__(
            self,
            outcomes=['done','preempted']
        )
        self._ac_client = ac_client
        self.stop_time = stop_time
        self.condition_func = condition_func

    def execute(self, userdata):
        rospy.loginfo(f"[StopState] Enter: STOP mode, stop_time={self.stop_time}, has_condition={self.condition_func is not None}")

        # 1) STOP Goal 전송
        stop_goal = ControlGoal(mode="STOP")
        self._ac_client.send_goal(stop_goal)

        start_time = rospy.Time.now()
        rate = rospy.Rate(10)  # 10Hz

        while not rospy.is_shutdown():
            if self.preempt_requested():
                rospy.logwarn("[StopState] Preempt requested => 'preempted'")
                self.service_preempt()
                self._ac_client.cancel_goal()
                return 'preempted'

            elapsed = (rospy.Time.now() - start_time).to_sec()
            time_done = False
            if self.stop_time is not None and elapsed >= self.stop_time:
                time_done = True

            cond_done = False
            if self.condition_func is not None and self.condition_func():
                cond_done = True

            if time_done or cond_done:
                rospy.loginfo("[StopState] stop wait ended. time_done=%s, cond_done=%s", time_done, cond_done)
                self._ac_client.cancel_goal()
                return 'done'

            state = self._ac_client.get_state()
            if state in [GoalStatus.ABORTED, GoalStatus.REJECTED]:
                rospy.logwarn("[StopState] STOP Action ended unexpectedly (state=%s). Still waiting though.", state)

            rate.sleep()
