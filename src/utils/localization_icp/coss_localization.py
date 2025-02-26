from crop_map import zoom_in_on_region
from BEV import convert_bev
from localization_ICP import extract_points_from_image, rescale_points, icp

import matplotlib.pyplot as plt
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge

class LocalizationICP:
    def __init__(self, baudrate=115200):
        # ROS 노드 생성 및 토픽 subscribe
        rospy.init_node("localization_node", anonymous=True)

        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.camCallback)
        self.odom_sub = rospy.Subscriber('',Float32, self.odomCallback)
        self.imu_sub = rospy.Subscriber('', Float32, self.imuCallback)

        # Camera 데이터 처리
        self.bridge = CvBridge()
        self.camera_img = None

    def camCallback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.camera_img = cv2.resize(frame, (480, 270))

    def odomCallback(self, msg):
        pass

    def imuCallback(self, msg):
        pass

    def run(self):
        # odemetry 위치 기반 지도 상의 ROI 추출
        map_path = "./test_img/SEAME_map.png"
        odom = [720, 425]
        imu_heading = 0
        
        map_roi, map_matrix = zoom_in_on_region(map_path, x = odom[0], y = odom[1], heading = imu_heading)
        if map_roi is None:
            raise ValueError("Failed to extract map ROI")
        print("map_roi shape:", map_roi.shape)

        # map_matrix 3x3 행렬로 변환
        map_matrix_3x3 = np.eye(3)
        map_matrix_3x3[:2, :] = map_matrix

        # odom을 ROI 좌표계로 변환
        odom_h = np.array([odom[0], odom[1], 1])
        odom_roi = map_matrix_3x3 @ odom_h

        rospy.sleep(1)
        # self.camera_img = "./test_img/mask1.jpg"
        if self.camera_img is None:
            print("No camera image received")
            return
        
        # BEV 이미지 추출
        bev_image = convert_bev(self.camera_img)
        print("bev_image shape:", bev_image.shape)

        # ICP localization
        # 점군 추출
        bev_points = extract_points_from_image(bev_image)
        map_points = extract_points_from_image(map_roi)
        
        print(f"BEV 점군 픽셀 개수: {len(bev_points)}")
        print(f"맵 점군 픽셀 개수: {len(map_points)}")
        
        scale = 0.75
        bev_points_phys = rescale_points(bev_points, scale)
        map_points_phys = map_points.copy()

        plt.figure("fig1",figsize=(10, 10))
        plt.scatter(bev_points_phys[:, 0], bev_points_phys[:, 1], c='red', label='Aligned BEV Points', s=1)
        plt.scatter(map_points_phys[:, 0], map_points_phys[:, 1], c='blue', label='Map Points', s=1)
        plt.title('before')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)

        # ICP 실행
        T_total, aligned_bev_points, final_error = icp(bev_points_phys, map_points_phys)
        print("[ICP] 최종 변환 행렬 (T_total):\n", T_total)
        print("[ICP] 최종 평균 매칭 오차:", final_error)

        # odom 위치 보정
        odom_h = np.array([odom[0],odom[1], 1])
        corrected_odom_roi = T_total @ odom_roi

        map_matrix_inv = np.linalg.inv(map_matrix_3x3)
        corrected_odom = map_matrix_inv @ corrected_odom_roi
        corrected_odom = corrected_odom[:2]

        print("Odometry correction: ", corrected_odom)
        
        # 변환 후 점군 시각화
        plt.figure("fig2",figsize=(10, 10))
        plt.scatter(aligned_bev_points[:, 0], aligned_bev_points[:, 1], c='red', label='Aligned BEV Points', s=1)
        plt.scatter(map_points_phys[:, 0], map_points_phys[:, 1], c='blue', label='Map Points', s=1)
        plt.title('after')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()

        rospy.spin()
        

if __name__ == '__main__':
    localization_process = LocalizationICP()
    # localization_process.start()
    # localization_process.join()
    localization_process.run()

