# #!/usr/bin/env python3

import rospy
import tf
import math
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Float32
from geometry_msgs.msg import Quaternion
import tf.transformations as transformations
from bfmc.msg import realsense_imu, bfmc_imu

# 초기 위치 및 속도 변수
# x_ = 11.77
# y_ = 2.05
# x_ = 0.3  # 30pixel
# y_ = 0.55 #5.45 # 55pixel

x_ = 0.45  # 30pixel
y_ = 0.55 

heading = 0.0 
linear_velocity_ = 0.0
angular_velocity_ = 0.0
prev_speed = 0.0
prev_heading = 0.0
quaternion = [0,0,0,1]
corrected_odom = [None, None]
local_update = False
odom_feedback = 0

# 퍼블리셔 초기화
odom_pub_ = None
yaw_pub_ = None
current_time_ = None
last_time_ = None

start_time = None
finish_time = None

def get_yaw_from_quaternion(quat):
    """쿼터니언 데이터를 사용하여 yaw 값(라디안)을 반환"""
    orientation_list = [quat.x, quat.y, quat.z, quat.w]
    (roll, pitch, yaw) = transformations.euler_from_quaternion(orientation_list)
    return yaw

def realsense_callback(msg):
    global angular_velocity_
    global heading
    global prev_heading
    global quaternion
    angular_velocity_ = msg.angular_velocity.z # get angular_velocity data
    heading = msg.yaw # get yaw data
    quaternion[0] = msg.orientation.x
    quaternion[1] = msg.orientation.y
    quaternion[2] = msg.orientation.z
    quaternion[3] = msg.orientation.w

def bfmc_callback(msg):
    global angular_velocity_,linear_velocity_, heading, prev_heading
    global quaternion
    global start_time, finish_time


    heading = (msg.yaw / 31635) * 360  # get yaw data
    yaw_pub_.publish(heading)
    if heading >180:
        yaw = heading-360
    else:
        yaw = heading
    heading = heading * math.pi / 180


    start_time = rospy.Time.now()
    if finish_time is not None:
        dt = (start_time - finish_time).to_sec()
        delta_yaw = heading - prev_heading
        angular_velocity_ = delta_yaw / dt # get angular_velocity data
        #linear_velocity_ = accel_x * dt # get linear_velocity data
    else:
        angular_velocity_ = 0.0
        #linear_velocity_ = 0.0

    
    finish_time = start_time
    prev_heading = heading

    quaternion = transformations.quaternion_from_euler(0, 0, -yaw*3.141592 / 180)
    #print(quaternion[2])

def speed_callback(msg):
    """속도 데이터 콜백 함수"""
    global linear_velocity_
    linear_velocity_ = msg.data / 100  # cm/s -> m/s 변환
    print("speed_callback",linear_velocity_)

def odom_callback(msg):
    global x_, y_, local_update, corrected_odom, odom_feedback
    corrected_x = msg.pose.pose.position.x
    corrected_y = msg.pose.pose.position.y

    if odom_feedback % 10 == 0:
        if abs(x_ - corrected_x) > 0.15 and abs(y_ - corrected_y) > 0.15:
            local_update = False
        else:
            local_update = True
            x_ = msg.pose.pose.position.x
            y_ = msg.pose.pose.position.y

    odom_feedback += 1

def update_odometry():
    """오도메트리 데이터 업데이트 및 퍼블리시"""
    global x_, y_, heading, current_time_, last_time_, corrected_odom, linear_velocity_

    current_time_ = rospy.Time.now()
    dt = (current_time_ - last_time_).to_sec()
    last_time_ = current_time_

    dx = (linear_velocity_) * math.cos(heading) * dt
    dy = (linear_velocity_) * math.sin(heading) * dt
    x_ += dx
    y_ -= dy

    # 오도메트리 메시지 생성
    odom = Odometry()
    odom.header.stamp = current_time_
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

    # TF 변환 브로드캐스트
    br = tf.TransformBroadcaster()
    br.sendTransform(
        (x_, y_, 0.0),
        quaternion,
        current_time_,
        "base_link",
        "odom"
    )


def main():
    global odom_pub_, current_time_, last_time_, yaw_pub_

    rospy.init_node('ini_odom_node', anonymous=True)
    rospy.loginfo("initial odom Node Started")



    # 퍼블리셔 및 서브스크라이버 설정
    odom_pub_ = rospy.Publisher('/odom', Odometry, queue_size=10)
    yaw_pub_ = rospy.Publisher('/realworld_yaw', Float64, queue_size=10)
    #rospy.Subscriber('/realsense_imu', realsense_imu, realsense_callback)
    rospy.Subscriber('/BFMC_imu', bfmc_imu, bfmc_callback)

    rospy.Subscriber('/sensor/speed', Float32, speed_callback)
    # rospy.Subscriber('/speed', Float32, speed_callback)
    rospy.Subscriber('localization/correctedOdom', Odometry, odom_callback)

    current_time_ = rospy.Time.now()
    last_time_ = current_time_

    rate = rospy.Rate(2)  # 50Hz 업데이트
    while not rospy.is_shutdown():
        update_odometry()
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

