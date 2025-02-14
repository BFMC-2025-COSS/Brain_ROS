#!/usr/bin/env python3

import rospy
import smach
import actionlib

from control.msg import ControlAction, ControlGoal
from actionlib_msgs.msg import GoalStatus

class UrbanState(smach.State):
    def __init__(self, ac_client, topic_data, range_data):
        """
        ac_client: actionlib.SimpleActionClient(ControlAction)
        get_sensor_func: 보행자/고속도로 감지 등의 외부 함수를 주입받을 수도 있음.
        """
        smach.State.__init__(
            self,
            outcomes=['highway_state', 'preempted']
        )
        self._ac_client = ac_client
        self.topic_data = topic_data
        self.range_data = range_data

    def execute(self, userdata):
        rospy.loginfo("[UrbanState] Enter state: URBAN driving")

        # 1) 액션 Goal 전송 (무기한 주행)
        goal = ControlGoal(mode="URBAN")
        self._ac_client.send_goal(goal)
        rospy.loginfo("[UrbanState] Sent goal: URBAN mode. Now indefinite driving...")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            # 2) 센서 또는 이벤트 체크
            # sensor_data = self._get_sensor_func()
            # if sensor_data['pedestrian']:
            #     rospy.loginfo("[UrbanState] Pedestrian detected -> transition")
            #     self._ac_client.cancel_goal()  # 현재 URBAN 주행 취소
            #     return 'pedestrian_detected'
            # if sensor_data['highway']:
            #     rospy.loginfo("[UrbanState] Highway entrance detected -> transition")
            #     self._ac_client.cancel_goal()
            #     return 'highway_state'
            try:
                # rospy.loginfo(f"{self.topic_data['closest_index']}, {self.range_data['highway'][0]}")
                if self.range_data['highway'][0] <= self.topic_data['closest_index'] < self.range_data['highway'][1]:
                    rospy.loginfo("[UrbanState] Highway entrance detected -> transition")
                    self._ac_client.cancel_goal()
                    return 'highway_state'
                
                
            except:
                pass

            # 3) preempt 체크 (상위에서 취소 명령이 들어왔나?)
            if self.preempt_requested():
                self.service_preempt()
                self._ac_client.cancel_goal()
                return 'preempted'

            # 4) 액션 서버 상태 확인 (실제론 indefinite이므로, SUCCEEDED 잘 안 뜸)
            #    하지만 혹시 서버 쪽에서 예외로 set_succeeded()나 set_aborted() 등을 호출할 수도 있으니 체크
            state = self._ac_client.get_state()
            if state in [GoalStatus.ABORTED, GoalStatus.REJECTED]:
                rospy.logwarn("[UrbanState] Action ended unexpectedly (state=%s).", state)
                return 'preempted'
            elif state == GoalStatus.SUCCEEDED:
                rospy.loginfo("[UrbanState] Action ended with success (unexpected?).")
                # 무기한 주행에서 서버가 종료될 이유가 없다면, 그냥 'preempted' 취급
                # 또는 'highway_state' 등 다른 전이로 정의 가능
                return 'preempted'

            rate.sleep()

        # 노드가 종료
        self._ac_client.cancel_goal()
        return 'preempted'
