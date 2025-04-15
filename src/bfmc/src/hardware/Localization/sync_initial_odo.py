#!/usr/bin/env python3
import rospy
import tf
import math
import message_filters
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64
import tf.transformations as transformations
from bfmc.msg import bfmc_imu  # bfmc_imu에는 orientation, yaw 필드가 있다고 가정

# === 전역 변수 초기화 ===
# Crop map
x_ = 0.3 
y_ = 5.45 

# RVIZ
# x_ = 0.45
# y_ = 0.55
heading = 0.0
prev_heading = 0.0
linear_velocity_ = 0.0
angular_velocity_ = 0.0
quaternion = [0, 0, 0, 1]
last_time_ = None

# 퍼블리셔
odom_pub_ = None
yaw_pub_ = None

# === 콜백: IMU + Speed 동기화 ===
def sync_callback(imu_msg, speed_msg):
    global x_, y_, heading, prev_heading, quaternion
    global linear_velocity_, angular_velocity_, odom_pub_, last_time_
    global yaw_pub_

    current_time = imu_msg.header.stamp

    # 초기 프레임
    if last_time_ is None:
        last_time_ = current_time
        return

    dt = (current_time - last_time_).to_sec()
    last_time_ = current_time

    # === Yaw 계산 ===
    heading_deg = (imu_msg.yaw / 31635) * 360.0
    yaw_pub_.publish(heading_deg)

    yaw_rad = math.radians(heading_deg - 360 if heading_deg > 180 else heading_deg)
    heading = yaw_rad

    # === 각속도 계산 ===
    delta_yaw = heading - prev_heading
    angular_velocity_ = delta_yaw / dt
    prev_heading = heading

    # === 선속도 계산 ===
    linear_velocity_ = speed_msg.twist.linear.x / 100.0  # m/s

    # === 위치 누적 ===
    dx = linear_velocity_ * math.cos(heading) * dt
    dy = linear_velocity_ * math.sin(heading) * dt
    x_ += dx
    y_ += dy

    # === 쿼터니언 ===
    quaternion = transformations.quaternion_from_euler(0, 0, -heading)

    # === Odometry 메시지 생성 ===
    odom = Odometry()
    odom.header.stamp = current_time
    odom.header.frame_id = "odom"
    odom.child_frame_id = "base_link"

    odom.pose.pose.position.x = x_
    odom.pose.pose.position.y = y_
    odom.pose.pose.position.z = 0.0

    odom.pose.pose.orientation.x = quaternion[0]
    odom.pose.pose.orientation.y = quaternion[1]
    odom.pose.pose.orientation.z = quaternion[2]
    odom.pose.pose.orientation.w = quaternion[3]

    odom.twist.twist.linear.x = linear_velocity_
    odom.twist.twist.angular.z = angular_velocity_

    odom_pub_.publish(odom)

    # === TF Broadcast ===
    br = tf.TransformBroadcaster()
    br.sendTransform(
        (x_, y_, 0.0),
        quaternion,
        current_time,
        "base_link",
        "odom"
    )

# === 메인 ===
def main():
    global odom_pub_, yaw_pub_

    rospy.init_node('sync_odom_node')
    rospy.loginfo("🛰️ sync_odom_node launched")

    odom_pub_ = rospy.Publisher('/odom', Odometry, queue_size=10)
    yaw_pub_ = rospy.Publisher('/realworld_yaw', Float64, queue_size=10)

    # === 동기화 서브스크라이버 ===
    imu_sub = message_filters.Subscriber('/BFMC_imu', bfmc_imu)
    speed_sub = message_filters.Subscriber('/sensor/speed', TwistStamped)

    # === Approximate 동기화 ===
    ts = message_filters.ApproximateTimeSynchronizer(
        [imu_sub, speed_sub],
        queue_size=10,
        slop=0.1
    )
    ts.registerCallback(sync_callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass



# #!/usr/bin/env python3
# import rospy
# import tf
# import math
# import message_filters
# from sensor_msgs.msg import Imu
# from nav_msgs.msg import Odometry
# from std_msgs.msg import Float64, Float32
# from geometry_msgs.msg import Quaternion, TwistStamped
# import tf.transformations as transformations
# from bfmc.msg import bfmc_imu

# # === 전역 변수 ===
# x_ = 0.45  # 초기 위치
# y_ = 0.55
# heading = 0.0
# linear_velocity_ = 0.0
# angular_velocity_ = 0.0
# prev_heading = 0.0
# quaternion = [0, 0, 0, 1]
# current_time_ = None
# last_time_ = None
# start_time = None
# finish_time = None
# odom_feedback = 0
# local_update = False

# # === 퍼블리셔 ===
# odom_pub_ = None
# yaw_pub_ = None

# # === 유틸 함수 ===
# def get_yaw_from_quaternion(quat):
#     """쿼터니언을 yaw로 변환"""
#     orientation_list = [quat.x, quat.y, quat.z, quat.w]
#     (_, _, yaw) = transformations.euler_from_quaternion(orientation_list)
#     return yaw

# # === 동기화된 콜백 ===
# def sync_callback(imu_msg, speed_msg):
#     global heading, prev_heading, linear_velocity_, angular_velocity_, quaternion
#     global start_time, finish_time, yaw_pub_

#     # === yaw 계산 (도 → 라디안) ===
#     heading_deg = (imu_msg.yaw / 31635) * 360
#     yaw_pub_.publish(heading_deg)

#     if heading_deg > 180:
#         yaw_deg = heading_deg - 360
#     else:
#         yaw_deg = heading_deg

#     heading = yaw_deg * math.pi / 180  # rad로 변환

#     # === angular_velocity 계산 ===
#     start_time = rospy.Time.now()
#     if finish_time is not None:
#         dt = (start_time - finish_time).to_sec()
#         delta_yaw = heading - prev_heading
#         angular_velocity_ = delta_yaw / dt
#     else:
#         angular_velocity_ = 0.0
#     finish_time = start_time
#     prev_heading = heading

#     # === quaternion 생성 ===
#     quaternion = transformations.quaternion_from_euler(0, 0, -yaw_deg * math.pi / 180)

#     # === linear_velocity 업데이트 ===
#     linear_velocity_ = speed_msg.twist.linear.x / 100.0  # cm/s → m/s

#     rospy.loginfo(f"[SYNC] speed={linear_velocity_:.3f} m/s, yaw={yaw_deg:.2f}°, dt={dt:.3f}")

# # === 오도메트리 계산 및 publish ===
# def update_odometry():
#     global x_, y_, heading, current_time_, last_time_, quaternion
#     global linear_velocity_, angular_velocity_, odom_pub_

#     current_time_ = rospy.Time.now()
#     dt = (current_time_ - last_time_).to_sec()
#     last_time_ = current_time_

#     dx = linear_velocity_ * math.cos(heading) * dt
#     dy = linear_velocity_ * math.sin(heading) * dt
#     x_ += dx
#     y_ -= dy

#     # === Odometry 메시지 생성 ===
#     odom = Odometry()
#     odom.header.stamp = current_time_
#     odom.header.frame_id = "odom"
#     odom.child_frame_id = "base_link"

#     odom.pose.pose.position.x = x_
#     odom.pose.pose.position.y = y_
#     odom.pose.pose.position.z = 0.0
#     odom.pose.pose.orientation.x = quaternion[0]
#     odom.pose.pose.orientation.y = quaternion[1]
#     odom.pose.pose.orientation.z = quaternion[2]
#     odom.pose.pose.orientation.w = quaternion[3]
#     odom.twist.twist.linear.x = linear_velocity_
#     odom.twist.twist.angular.z = angular_velocity_

#     odom_pub_.publish(odom)

#     # === TF 브로드캐스트 ===
#     br = tf.TransformBroadcaster()
#     br.sendTransform(
#         (x_, y_, 0.0),
#         quaternion,
#         current_time_,
#         "base_link",
#         "odom"
#     )

# # === 메인 함수 ===
# def main():
#     global odom_pub_, yaw_pub_, current_time_, last_time_

#     rospy.init_node('sync_odom_node')
#     rospy.loginfo("🛰️ sync_odom_node started")

#     # === 퍼블리셔 ===
#     odom_pub_ = rospy.Publisher('/odom', Odometry, queue_size=10)
#     yaw_pub_ = rospy.Publisher('/realworld_yaw', Float64, queue_size=10)

#     # === 동기화 서브스크라이버 ===
#     imu_sub = message_filters.Subscriber('/BFMC_imu', bfmc_imu)
#     speed_sub = message_filters.Subscriber('/sensor/speed', TwistStamped)

#     ts = message_filters.ApproximateTimeSynchronizer(
#         [imu_sub, speed_sub],
#         queue_size=10,
#         slop=0.1
#     )
#     ts.registerCallback(sync_callback)

#     # === 초기 시간 설정 ===
#     current_time_ = rospy.Time.now()
#     last_time_ = current_time_

#     rate = rospy.Rate(2)  # 2Hz
#     while not rospy.is_shutdown():
#         update_odometry()
#         # rate.sleep()

# if __name__ == '__main__':
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass

