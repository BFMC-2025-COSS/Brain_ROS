# from crop_map import zoom_in_on_region
# from BEV import convert_bev
# from localization_ICP import extract_points_from_image, rescale_points, icp

# import matplotlib.pyplot as plt
# import rospy
# import cv2
# import numpy as np
# from sensor_msgs.msg import Image
# from nav_msgs.msg import Odometry
# from std_msgs.msg import Float32, String
# from bfmc.msg import bfmc_imu
# from cv_bridge import CvBridge
# import message_filters

# class LocalizationICP:
#     def __init__(self, baudrate=115200):
#         # ROS 노드 생성 및 토픽 subscribe
#         rospy.init_node("localization_node", anonymous=True)

#         #self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.camCallback)
#         self.mask_sub = rospy.Subscriber('/camera/lane_mask', Image, self.maskCallback)
#         self.odom_sub = rospy.Subscriber('/odom',Odometry, self.odomCallback)
#         self.imu_sub = rospy.Subscriber('/BFMC_imu', bfmc_imu, self.imuCallback)

#         self.odom_pub = rospy.Publisher('/localization/correctedOdom', Odometry, queue_size=10)

#         # Camera 데이터 처리
#         self.bridge = CvBridge()
#         self.camera_img = None
#         self.mask_img = None

#         # Odometry
#         self.odom = [None, None]
#         self.heading = None

#         self.map_img = cv2.imread("./test_img/SEAME_map.png")

#     def camCallback(self, msg):
#         frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
#         self.camera_img = cv2.resize(frame, (480, 270))

#     def maskCallback(self, msg):
#         frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
#         self.mask_img = cv2.resize(frame, (480, 270))
    
#     def odomCallback(self, msg):
#         self.odom[0] = int(msg.pose.pose.position.x * 100)
#         self.odom[1] = int(msg.pose.pose.position.y * 100)
#         # print("Initial odometry: ", self.odom)
#         pass

#     def imuCallback(self, msg):
#         self.heading = (-(msg.yaw / 31635) * 360 + 270) % 360
#         print("Heading value: ", self.heading)
    
#     def cropped_to_global(self, Xc,Yc, X0, Y0, theta):
#         Xc = Xc / 3 - 80 #+ 17
#         Yc = Yc / 3 - 90 #- 16
#         theta_rad = np.radians(theta)
#         R = np.array([[np.cos(theta_rad), np.sin(theta_rad)], 
#                       [-np.sin(theta_rad), np.cos(theta_rad)]])
#         Xg, Yg = R @ np.array([Xc, Yc]) + np.array([X0, Y0])
#         return Xg, Yg

#     def conver_to_ros_coords(self, x_cv, y_cv, image_height):
#         x_ros = x_cv
#         y_ros = image_height - y_cv
#         return x_ros, y_ros

#     def run(self):
#         # odemetry 위치 기반 지도 상의 ROI 추출
        
#         self.odom = [420, 460]
#         self.heading = 0
        
#         print("START")
#         print(self.odom)
#         print("Zoom the map")

#         rate = rospy.Rate(2)

#         while not rospy.is_shutdown():
#             if self.odom[0] == None or self.heading == None:
#                 rospy.sleep(1)
#             print("="*100, "\nInit Odom:   ", self.odom, "\nHeading:    ", self.heading, "\n", "="*100)

#             T_total, aligned_bev_points, final_error = None, None, 5.0
#             x_roi,y_roi = self.odom[0], self.odom[1]

#             for t in range(0, 1):
#                 map_roi, map_matrix = zoom_in_on_region(self.map_img, x = self.odom[0] + (30 * np.sin(self.heading) * t), y = self.odom[1] + (30 * np.cos(self.heading) * t), heading = self.heading)
#                 if map_roi is None:
#                     raise ValueError("Failed to extract map ROI")
#                 print("map_roi shape:", map_roi.shape)

#                 # map_matrix 3x3 행렬로 변환
#                 map_matrix_3x3 = np.eye(3)
#                 map_matrix_3x3[:2, :] = map_matrix

#                 # odom을 ROI 좌표계로 변환
#                 odom_h = np.array([self.odom[0], self.odom[1], 1])
#                 odom_roi = map_matrix_3x3 @ odom_h

#                 # if self.camera_img is None:
#                 #     print("No camera image received")
#                 #     return
#                 if self.mask_img is None:
#                     print("No mask image received")
#                     continue
                
#                 # BEV 이미지 추출
#                 # bev_image = convert_bev(self.camera_img)
#                 bev_image = convert_bev(self.mask_img)
#                 #print("bev_image shape:", bev_image.shape)

#                 # ICP localization
#                 # 점군 추출
#                 bev_points = extract_points_from_image(bev_image)
#                 map_points = extract_points_from_image(map_roi)
                
#                 print(f"BEV 점군 픽셀 개수: {len(bev_points)}")
#                 print(f"맵 점군 픽셀 개수: {len(map_points)}")

#                 if len(bev_points) == 0 or len(map_points) == 0:
#                     print("="*100,"\nAny point is detected\n","="*100)
#                     continue
                
#                 scale = 0.75
#                 bev_points_phys = rescale_points(bev_points, scale)
#                 map_points_phys = map_points.copy()

#                 bev_view = rescale_points(bev_points_phys, 5)
#                 map_view = rescale_points(map_points_phys, 5)

#                 # ICP 실행
#                 T_total_example, aligned_bev_points_example, final_error_example = icp(bev_points_phys, map_points_phys)

#                 if final_error_example < final_error:
#                     T_total = T_total_example
#                     aligned_bev_points = aligned_bev_points_example
#                     final_error = final_error_example
#                     x_roi,y_roi = self.odom[0] + (30 * np.sin(self.heading) * t), self.odom[1] + (30 * np.cos(self.heading) * t)

#                     if final_error < 1:
#                         break

#             # map_roi, map_matrix = zoom_in_on_region(self.map_img, x = self.odom[0], y = self.odom[1], heading = self.heading)
#             # if map_roi is None:
#             #     raise ValueError("Failed to extract map ROI")
#             # print("map_roi shape:", map_roi.shape)

#             # # map_matrix 3x3 행렬로 변환
#             # map_matrix_3x3 = np.eye(3)
#             # map_matrix_3x3[:2, :] = map_matrix

#             # # odom을 ROI 좌표계로 변환
#             # odom_h = np.array([self.odom[0], self.odom[1], 1])
#             # odom_roi = map_matrix_3x3 @ odom_h

#             # # if self.camera_img is None:
#             # #     print("No camera image received")
#             # #     return
#             # if self.mask_img is None:
#             #     print("No mask image received")
#             #     continue
            
#             # # BEV 이미지 추출
#             # # bev_image = convert_bev(self.camera_img)
#             # bev_image = convert_bev(self.mask_img)
#             # #print("bev_image shape:", bev_image.shape)

#             # # ICP localization
#             # # 점군 추출
#             # bev_points = extract_points_from_image(bev_image)
#             # map_points = extract_points_from_image(map_roi)
            
#             # print(f"BEV 점군 픽셀 개수: {len(bev_points)}")
#             # print(f"맵 점군 픽셀 개수: {len(map_points)}")

#             # if len(bev_points) == 0 or len(map_points) == 0:
#             #     print("="*100,"\nAny point is detected\n","="*100)
#             #     continue
            
#             # scale = 0.75
#             # bev_points_phys = rescale_points(bev_points, scale)
#             # map_points_phys = map_points.copy()

#             # bev_view = rescale_points(bev_points_phys, 5)
#             # map_view = rescale_points(map_points_phys, 5)

#             # # ICP 실행
#             # T_total, aligned_bev_points, final_error = icp(bev_points_phys, map_points_phys)
#             #print("[ICP] 최종 변환 행렬 (T_total):\n", T_total)
#             print("[ICP] 최종 평균 매칭 오차:", final_error)

#             canvas = np.ones((500, 500, 3), dtype=np.uint8)
#             if bev_view is not None and map_view is not None:
#                 for point in bev_view:
#                     x, y = int(point[0]), int(point[1])
#                     if 0 <= x < 500 and 0 <= y < 500:
#                         cv2.circle(canvas, (x, y), 1, (0, 0, 255), -1)

#                 for point in map_view:
#                     x, y = int(point[0]), int(point[1])
#                     if 0 <= x < 500 and 0 <= y < 500:
#                         cv2.circle(canvas, (x, y), 1, (255, 0, 0), -1)

#             # odom 위치 보정
#             # odom_h = np.array([self.odom[0],self.odom[1], 1])
#             odom_h = np.array([x_roi,y_roi, 1])
#             corrected_odom_roi = T_total @ odom_h

#             map_matrix_inv = np.linalg.inv(map_matrix_3x3)
#             corrected_odom = np.linalg.inv(map_matrix_inv) @ corrected_odom_roi
#             #corrected_odom = corrected_odom[:2]
#             #corrected_odom = np.linalg.inv(T_total) @ [240,295,1] 
#             corrected_odom = np.linalg.inv(T_total) @ [260,255,1]
#             xg,yg = self.cropped_to_global(corrected_odom[0], corrected_odom[1], self.odom[0], self.odom[1], self.heading)
#             print("Corrected odom: ", xg, yg)
#             corrected_odom = [xg, yg]
            
#             odom_msg = Odometry()
#             odom_msg.header.stamp = rospy.Time.now()
#             odom_msg.header.frame_id = "odom"
#             odom_msg.child_frame_id = "base_link"
            
#             if final_error < 3:
#                 odom_msg.pose.pose.position.x = corrected_odom[0] / 100
#                 odom_msg.pose.pose.position.y = corrected_odom[1] / 100
#                 odom_msg.pose.pose.position.z = 0
#             else:
#                 odom_msg.pose.pose.position.x = self.odom[0] / 100
#                 odom_msg.pose.pose.position.y = self.odom[1] / 100
#                 odom_msg.pose.pose.position.z = 0
            
#             self.odom_pub.publish(odom_msg)

#             print("Odometry correction: ", corrected_odom)

#             aligned_view = rescale_points(aligned_bev_points, 5)

#             for point in aligned_view:
#                 x, y = int(point[0]), int(point[1])
#                 if 0 <= x < 500 and 0 <= y < 500:
#                     cv2.circle(canvas, (x, y), 1, (0, 255, 0), -1)

#             cv2.imshow("ICP Localization", canvas)
#             cv2.waitKey(1)

#             rate.sleep()

#         # rospy.spin()
        

# if __name__ == '__main__':
#     localization_process = LocalizationICP()
#     #localization_process.start()
#     # localization_process.join()
#     localization_process.run()
#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import tf
import message_filters
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String
from bfmc.msg import bfmc_imu
from cv_bridge import CvBridge
from crop_map import zoom_in_on_region
from BEV import convert_bev
from localization_ICP import extract_points_from_image, rescale_points, icp
import time

class LocalizationICP:
    def __init__(self):
        rospy.init_node("localization_node", anonymous=True)

        self.imu_sub = rospy.Subscriber('/BFMC_imu', bfmc_imu, self.imuCallback)
        self.odom_pub = rospy.Publisher('/localization/correctedOdom', Odometry, queue_size=10)

        self.bridge = CvBridge()
        self.mask_img = None
        self.heading = None
        self.odom = [None, None]
        self.start = 0

        self.map_img = cv2.imread("/home/seame/Brain_ROS/src/bfmc/src/hardware/Localization/test_img/SEAME_map.png")

    def imuCallback(self, msg):
        self.heading = (-(msg.yaw / 31635) * 360 + 270) % 360
        # rospy.loginfo(f"IMU heading: {self.heading:.2f}°")

    def cropped_to_global(self, Xc, Yc, X0, Y0, theta):
        Xc = Xc / 3 - 80
        Yc = Yc / 3 - 90
        theta_rad = np.radians(theta)
        R = np.array([[np.cos(theta_rad), np.sin(theta_rad)],
                      [-np.sin(theta_rad), np.cos(theta_rad)]])
        Xg, Yg = R @ np.array([Xc, Yc]) + np.array([X0, Y0])
        return Xg, Yg

    def sync_callback(self, odom_msg, mask_msg):
        rospy.logwarn("sync_callback is working")
        if self.heading is None:
            rospy.logwarn("IMU heading 값이 아직 없음.")
            return
        self.odom = [
            int(odom_msg.pose.pose.position.x * 100),
            int(odom_msg.pose.pose.position.y * 100)
        ]
        self.mask_img = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding="mono8")
        self.mask_img = cv2.resize(self.mask_img, (480, 270))
        print("sync_callback")
        self.process_localization_step()

    def process_localization_step(self):
        x_roi, y_roi = self.odom[0], self.odom[1]
        heading = self.heading
        T_total, aligned_bev_points, final_error = None, None, 5.0

        for t in range(1):
            map_roi, map_matrix = zoom_in_on_region(self.map_img,
                                                    x=self.odom[0] + (30 * np.sin(heading) * t),
                                                    y=self.odom[1] + (30 * np.cos(heading) * t),
                                                    heading=heading)
            if map_roi is None:
                rospy.logwarn("Failed to extract map ROI")
                return
            
            # map_matrix 3x3 행렬로 변환
            map_matrix_3x3 = np.eye(3)
            map_matrix_3x3[:2, :] = map_matrix

            if self.mask_img is None:
                    print("No mask image received")
                    return

            # odom을 ROI 좌표계로 변환
            odom_h = np.array([self.odom[0], self.odom[1], 1])
            odom_roi = map_matrix_3x3 @ odom_h

            # BEV 이미지 추출
            bev_image = convert_bev(self.mask_img)

            # 점군 추출
            bev_points = extract_points_from_image(bev_image)
            map_points = extract_points_from_image(map_roi)

            print(f"BEV 점군 픽셀 개수: {len(bev_points)}")
            print(f"맵 점군 픽셀 개수: {len(map_points)}")

            if len(bev_points) == 0 or len(map_points) == 0:
                rospy.logwarn("Point set is empty.")
                return

            bev_points_phys = rescale_points(bev_points, 0.75)
            map_points_phys = map_points.copy()

            bev_view = rescale_points(bev_points_phys, 5)
            map_view = rescale_points(map_points_phys, 5)

            T_test, aligned_pts, err = icp(bev_points_phys, map_points_phys)
            if err < final_error:
                T_total = T_test
                aligned_bev_points = aligned_pts
                final_error = err
                x_roi = self.odom[0] + (30 * np.sin(heading) * t)
                y_roi = self.odom[1] + (30 * np.cos(heading) * t)

        rospy.loginfo(f"[ICP] 최종 평균 매칭 오차: {final_error:.3f}")

        if T_total is None:
            return

        # 시각화
        canvas = np.ones((500, 500, 3), dtype=np.uint8)
        if bev_view is not None and map_view is not None:
            for point in bev_view:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < 500 and 0 <= y < 500:
                    cv2.circle(canvas, (x, y), 1, (0, 0, 255), -1)

            for point in map_view:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < 500 and 0 <= y < 500:
                    cv2.circle(canvas, (x, y), 1, (255, 0, 0), -1)

        # odom 위치 보정
        odom_h = np.array([x_roi,y_roi, 1])
        corrected_odom_roi = T_total @ odom_h

        map_matrix_inv = np.linalg.inv(map_matrix_3x3)
        corrected_odom = np.linalg.inv(map_matrix_inv) @ corrected_odom_roi
        #corrected_odom = corrected_odom[:2]
        #corrected_odom = np.linalg.inv(T_total) @ [240,295,1] 
        corrected_odom = np.linalg.inv(T_total) @ [260,255,1]
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
        end = time.time()
        print("Odometry correction: ", corrected_odom)
        print("time: ", end - self.start)
        with open("time_log.txt", "a") as file:  # "a"는 append 모드로, 기존 내용 뒤에 덧붙이기
            file.write(f"{end - self.start}\n")
        self.start = time.time()
        

        

        aligned_view = rescale_points(aligned_bev_points, 5)

        for point in aligned_view:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < 500 and 0 <= y < 500:
                cv2.circle(canvas, (x, y), 1, (0, 255, 0), -1)
        #cv2.imshow("ICP Localization", canvas)
        #cv2.waitKey(1)
        self.latest_canvas = canvas

    def run(self):
        # def odom_debug(msg):
        #     rospy.loginfo(f"[Odom] {msg.header.stamp.to_sec()}")
        # def mask_debug(msg):
        #     rospy.loginfo(f"[Mask] {msg.header.stamp.to_sec()}")

        # rospy.Subscriber('/odom', Odometry, odom_debug)
        # rospy.Subscriber('/camera/lane_mask', Image, mask_debug)

        odom_sub = message_filters.Subscriber('/odom', Odometry)
        mask_sub = message_filters.Subscriber('/camera/lane_mask', Image)

        ts = message_filters.ApproximateTimeSynchronizer(
            [odom_sub, mask_sub],
            queue_size=50,
            slop=0.5
        )
        ts.registerCallback(self.sync_callback)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if hasattr(self, 'latest_canvas'):
                cv2.imshow("ICP Localization", self.latest_canvas)
                cv2.waitKey(1)
            rate.sleep()
        rospy.spin()

if __name__ == '__main__':
    loc = LocalizationICP()
    loc.run()

