from crop_map import zoom_in_on_region
from BEV import convert_bev
from localization_ICP import extract_points_from_image, rescale_points, icp

import matplotlib.pyplot as plt
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String
from bfmc.msg import bfmc_imu
from cv_bridge import CvBridge

class LocalizationICP:
    def __init__(self, baudrate=115200):
        # ROS 노드 생성 및 토픽 subscribe
        rospy.init_node("localization_node", anonymous=True)

        #self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.camCallback)
        self.mask_sub = rospy.Subscriber('/camera/lane_mask', Image, self.maskCallback)
        self.odom_sub = rospy.Subscriber('/odom',Odometry, self.odomCallback)
        self.imu_sub = rospy.Subscriber('/BFMC_imu', bfmc_imu, self.imuCallback)

        self.odom_pub = rospy.Publisher('/localization/correctedOdom', Odometry, queue_size=10)

        # Camera 데이터 처리
        self.bridge = CvBridge()
        self.camera_img = None
        self.mask_img = None

        # Odometry
        self.odom = [None, None]
        self.heading = None

        self.map_img = cv2.imread("./test_img/SEAME_map.png")

    def camCallback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.camera_img = cv2.resize(frame, (480, 270))

    def maskCallback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        self.mask_img = cv2.resize(frame, (480, 270))
    
    def odomCallback(self, msg):
        self.odom[0] = int(msg.pose.pose.position.x * 100)
        self.odom[1] = int(msg.pose.pose.position.y * 100)
        # print("Initial odometry: ", self.odom)
        pass

    def imuCallback(self, msg):
        self.heading = (-(msg.yaw / 31635) * 360 + 270) % 360
        # print("Heading value: ", self.heading)
    
    def cropped_to_global(self, Xc,Yc, X0, Y0, theta):
        Xc = Xc / 3 - 80 #+ 17
        Yc = Yc / 3 - 90 #- 16
        theta_rad = np.radians(theta)
        R = np.array([[np.cos(theta_rad), np.sin(theta_rad)], 
                      [-np.sin(theta_rad), np.cos(theta_rad)]])
        Xg, Yg = R @ np.array([Xc, Yc]) + np.array([X0, Y0])
        return Xg, Yg

    def conver_to_ros_coords(self, x_cv, y_cv, image_height):
        x_ros = x_cv
        y_ros = image_height - y_cv
        return x_ros, y_ros

    def run(self):
        # odemetry 위치 기반 지도 상의 ROI 추출
        
        self.odom = [30, 545]
        self.heading = 0
        
        print("START")
        print(self.odom)
        print("Zoom the map")

        while not rospy.is_shutdown():
            if self.odom[0] == None or self.heading == None:
                rospy.sleep(1)
            print("="*100, "\nInit Odom:   ", self.odom, "\nHeading:    ", self.heading, "\n", "="*100)

            map_roi, map_matrix = zoom_in_on_region(self.map_img, x = self.odom[0], y = self.odom[1], heading = self.heading)
            if map_roi is None:
                raise ValueError("Failed to extract map ROI")
            print("map_roi shape:", map_roi.shape)

            # map_matrix 3x3 행렬로 변환
            map_matrix_3x3 = np.eye(3)
            map_matrix_3x3[:2, :] = map_matrix

            # odom을 ROI 좌표계로 변환
            odom_h = np.array([self.odom[0], self.odom[1], 1])
            odom_roi = map_matrix_3x3 @ odom_h

            # if self.camera_img is None:
            #     print("No camera image received")
            #     return
            if self.mask_img is None:
                print("No mask image received")
                continue
            
            # BEV 이미지 추출
            # bev_image = convert_bev(self.camera_img)
            bev_image = convert_bev(self.mask_img)
            #print("bev_image shape:", bev_image.shape)

            # ICP localization
            # 점군 추출
            bev_points = extract_points_from_image(bev_image)
            map_points = extract_points_from_image(map_roi)
            
            #print(f"BEV 점군 픽셀 개수: {len(bev_points)}")
            #print(f"맵 점군 픽셀 개수: {len(map_points)}")

            if len(bev_points) == 0 or len(map_points) == 0:
                print("="*100,"\nAny point is detected\n","="*100)
                continue
            
            scale = 0.75
            bev_points_phys = rescale_points(bev_points, scale)
            map_points_phys = map_points.copy()

            bev_view = rescale_points(bev_points_phys, 5)
            map_view = rescale_points(map_points_phys, 5)

            canvas = np.ones((500, 500, 3), dtype=np.uint8)

            for point in bev_view:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < 500 and 0 <= y < 500:
                    cv2.circle(canvas, (x, y), 1, (0, 0, 255), -1)

            for point in map_view:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < 500 and 0 <= y < 500:
                    cv2.circle(canvas, (x, y), 1, (255, 0, 0), -1)

            # ICP 실행
            T_total, aligned_bev_points, final_error = icp(bev_points_phys, map_points_phys)
            #print("[ICP] 최종 변환 행렬 (T_total):\n", T_total)
            print("[ICP] 최종 평균 매칭 오차:", final_error)

            # odom 위치 보정
            odom_h = np.array([self.odom[0],self.odom[1], 1])
            corrected_odom_roi = T_total @ odom_h

            map_matrix_inv = np.linalg.inv(map_matrix_3x3)
            corrected_odom = np.linalg.inv(map_matrix_inv) @ corrected_odom_roi
            #corrected_odom = corrected_odom[:2]
            corrected_odom = np.linalg.inv(T_total) @ [240,295,1]
            xg,yg = self.cropped_to_global(corrected_odom[0], corrected_odom[1], self.odom[0], self.odom[1], self.heading)
            print("Corrected odom: ", xg, yg)
            corrected_odom = [xg, yg]
            
            odom_msg = Odometry()
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.header.frame_id = "odom"
            odom_msg.child_frame_id = "base_link"
            
            if final_error < 3:
                odom_msg.pose.pose.position.x = corrected_odom[0] / 100
                odom_msg.pose.pose.position.y = corrected_odom[1] / 100
                odom_msg.pose.pose.position.z = 0
            else:
                odom_msg.pose.pose.position.x = self.odom[0] / 100
                odom_msg.pose.pose.position.y = self.odom[1] / 100
                odom_msg.pose.pose.position.z = 0
            
            self.odom_pub.publish(odom_msg)

            print("Odometry correction: ", corrected_odom)

            aligned_view = rescale_points(aligned_bev_points, 5)

            for point in aligned_view:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < 500 and 0 <= y < 500:
                    cv2.circle(canvas, (x, y), 1, (0, 255, 0), -1)

            cv2.imshow("ICP Localization", canvas)
            cv2.waitKey(1)
        

if __name__ == '__main__':
    localization_process = LocalizationICP()
    localization_process.run()
