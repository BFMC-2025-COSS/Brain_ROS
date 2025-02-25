from crop_map import zoom_in_on_region
from BEV import convert_bev
from localization_ICP import extract_points_from_image, rescale_points, icp

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # odemetry 위치 기반 지도 상의 ROI 추출
    map_path = "./test_img/SEAME_map.png"
    odom = [720, 425]
    imu_heading = 0
    
    map_roi = zoom_in_on_region(map_path, x = odom[0], y = odom[1], heading = imu_heading)
    print("map_roi shape:", map_roi.shape)

    # BEV 이미지 추출
    camera_img = "./test_img/mask1.jpg"
    bev_image = convert_bev(camera_img)
    print("bev_image shape:", bev_image.shape)

    # ICP localization
    # 점군 추출
    bev_points = extract_points_from_image(bev_image)
    map_points = extract_points_from_image(map_roi)
    
    print(f"BEV 점군 픽셀 개수: {len(bev_points)}")
    print(f"맵 점군 픽셀 개수: {len(map_points)}")
    
    scale = 1
    
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
    
    # 변환 후 점군 시각화
    plt.figure("fig2",figsize=(10, 10))
    plt.scatter(aligned_bev_points[:, 0], aligned_bev_points[:, 1], c='red', label='Aligned BEV Points', s=1)
    plt.scatter(map_points_phys[:, 0], map_points_phys[:, 1], c='blue', label='Map Points', s=1)
    plt.title('after')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

