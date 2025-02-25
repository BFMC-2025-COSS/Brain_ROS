# import numpy as np
# import cv2
# from scipy.spatial import KDTree
# import matplotlib.pyplot as plt



# def extract_points_from_image(image_path, threshold_val=127):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Cannot load image: {image_path}")
#     # 그레이스케일 변환
#     if len(image.shape) == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image
#     ret, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
#     pts = cv2.findNonZero(binary)
#     if pts is None:
#         return np.empty((0, 2), dtype=np.float32)
#     pts = pts.reshape(-1, 2)
#     return pts.astype(np.float32)

# def rescale_points(points, scale):
#     return points * scale

# def nearest_neighbor(src, dst):
#     tree = KDTree(dst)
#     distances, indices = tree.query(src)
#     return distances, indices

# def compute_transform(src, dst):
#     centroid_src = np.mean(src, axis=0)
#     centroid_dst = np.mean(dst, axis=0)
    
#     src_centered = src - centroid_src
#     dst_centered = dst - centroid_dst
    
#     W = np.dot(dst_centered.T, src_centered)
#     U, _, Vt = np.linalg.svd(W)
#     R = np.dot(U, Vt)
    
#     # 회전행렬이 반사를 나타내면 수정
#     if np.linalg.det(R) < 0:
#         Vt[-1, :] *= -1
#         R = np.dot(U, Vt)
    
#     t = centroid_dst - np.dot(R, centroid_src)
#     return R, t

# def icp(source, target, max_iterations=20, tolerance=1e-6, num_samples=5000):
#     # src = random_sampling(source, num_samples)
#     # dst = random_sampling(target, num_samples)
#     src = source.copy()
#     dst = target.copy()
    
#     T_total = np.eye(3)
#     prev_error = float('inf')
    
#     for i in range(max_iterations):
#         print(f"Iteration: {i}")
#         distances, indices = nearest_neighbor(src, dst)
#         matched_dst = dst[indices]
        
#         R, t = compute_transform(src, matched_dst)
        
#         T = np.eye(3)
#         T[:2, :2] = R
#         T[:2, 2] = t
        
#         T_total = T @ T_total
        
#         # src_hom = np.hstack((src, np.ones((src.shape[0], 1))))
#         src_hom = np.hstack((src, np.ones((src.shape[0], 1), dtype=np.float32)))
#         src_transformed = (T @ src_hom.T).T
#         src = src_transformed[:, :2]
        
#         mean_error = np.mean(distances)
#         if abs(prev_error - mean_error) < tolerance:
#             break
#         prev_error = mean_error
        
#     return T_total, src, mean_error

# def warp_bev_to_map(bev_image, T_total, scale):
#     D = np.diag([scale, scale, 1])
#     D_inv = np.linalg.inv(D)
#     T_pixel = D_inv @ T_total @ D
#     T_affine = T_pixel[:2, :]  # cv2.warpAffine에 사용 가능한 2x3 변환 행렬
    
#     # BEV 이미지 크기를 가져와 워핑. (실제 적용 시 원하는 출력 크기에 맞춰 조정)
#     h, w = bev_image.shape[:2]
#     warped_bev = cv2.warpAffine(bev_image, T_affine, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
#     return warped_bev

# def random_sampling(points, num_samples):
#     indices = np.random.choice(points.shape[0], num_samples, replace=False)
#     return points[indices]

# if __name__ == '__main__':
#     # 파일 경로 (적절히 수정)
#     # bev_image_path = './bev_image.png'    # BEV로 추출한 차선 이미지
#     bev_image_path = './bev_image1.png'
#     # map_image_path = './resized_image.jpg'     # 원본 맵 사진
#     map_image_path = './zoom_img.png'     # 원본 맵 사진
    
#     # 이미지 로드
#     bev_image = cv2.imread(bev_image_path)
#     map_image = cv2.imread(map_image_path)
#     if bev_image is None or map_image is None:
#         raise ValueError("이미지를 불러올 수 없습니다.")
    
#     # 점군 추출
#     bev_points = extract_points_from_image(bev_image_path)
#     map_points = extract_points_from_image(map_image_path)
    
#     print(f"BEV 점군 픽셀 개수: {len(bev_points)}")
#     print(f"맵 점군 픽셀 개수: {len(map_points)}")
    
#     # 스케일 설정
#     scale = 1
#     bev_points_phys = rescale_points(bev_points, 1)
#     map_points_phys = rescale_points(map_points, scale)
#     # BEV 점군과 맵 점군 시각화
#     plt.figure(figsize=(10, 10))
#     plt.scatter(bev_points_phys[:, 0], bev_points_phys[:, 1], c='red', label='BEV Points', s=10)
#     plt.scatter(map_points_phys[:, 0], map_points_phys[:, 1], c='blue', label='Map Points', s=1)
#     plt.title('BEV Points vs Map Points')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.legend()
#     plt.axis('equal')
#     plt.grid(True)
#     plt.show()

#     # ICP 실행
#     T_total, aligned_bev_points, final_error = icp(bev_points_phys, map_points_phys)
#     print("누적 변환 행렬 (T_total):\n", T_total)
#     print("최종 평균 매칭 오차:", final_error)
    
#     # BEV 이미지를 맵 좌표계로 워핑
#     warped_bev = warp_bev_to_map(bev_image, T_total, scale)
#     if warped_bev is None:
#         raise ValueError("warpAffine 결과 warped_bev가 None입니다. T_affine 값 및 이미지 크기를 확인하세요.")

#     # warped_bev의 shape 확인
#     print("warped_bev shape:", warped_bev.shape)

#     # warped_bev가 흑백(1채널)일 때만 COLOR_GRAY2BGR 적용
#     if len(warped_bev.shape) == 2 or warped_bev.shape[2] == 1:
#         warped_bev_color = cv2.cvtColor(warped_bev, cv2.COLOR_GRAY2BGR)
#     else:
#         warped_bev_color = warped_bev

#     # 크기 조정
#     map_h, map_w = map_image.shape[:2]
#     warped_bev = cv2.resize(warped_bev, (map_w, map_h))

#     # 오버레이
#     map_image = cv2.resize(map_image, (warped_bev_color.shape[1], warped_bev_color.shape[0]))

#     print("map_image shape:", map_image.shape)  # (H, W, C) 형태
#     print("warped_bev_color shape:", warped_bev_color.shape)  # (H, W, C) 형태


#     print("map_image channels:", map_image.shape[-1] if len(map_image.shape) == 3 else 1)
#     print("warped_bev_color channels:", warped_bev_color.shape[-1] if len(warped_bev_color.shape) == 3 else 1)


#     overlay = cv2.addWeighted(map_image, 0.7, warped_bev_color, 0.3, 0)
#     cv2.imshow("Overlay of Warped BEV on Map", overlay)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import numpy as np
import cv2
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def extract_points_from_image(image_path, threshold_val=127):
    # image = cv2.imread(image_path)
    image = image_path
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    ret, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    pts = cv2.findNonZero(binary)
    if pts is None:
        return np.empty((0, 2), dtype=np.float32)

    pts = pts.reshape(-1, 2)
    return pts.astype(np.float32)

def rescale_points(points, scale):
    return points * scale

def nearest_neighbor(src, dst):
    tree = KDTree(dst)
    distances, indices = tree.query(src)
    return distances, indices

def compute_transform(src, dst):
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)
    
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst
    
    W = np.dot(dst_centered.T, src_centered)
    U, _, Vt = np.linalg.svd(W)
    R = np.dot(U, Vt)

    # 반사행렬(det < 0) 보정
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1 
        R = np.dot(U, Vt)
    
    t = centroid_dst - np.dot(R, centroid_src)
    return R, t

def icp(source, target, max_iterations=20, tolerance=1e-6):
    src = source.copy()
    dst = target.copy()

    T_total = np.eye(3, dtype=np.float32)
    prev_error = float('inf')
    
    for i in range(max_iterations):
        print(f"Iteration: {i}")
        distances, indices = nearest_neighbor(src, dst)
        matched_dst = dst[indices]
        
        R, t = compute_transform(src, matched_dst)
        
        # 현재 단계에서의 2D rigid 변환 행렬 만들기
        T = np.eye(3, dtype=np.float32)
        T[:2, :2] = R
        T[:2, 2] = t
        
        # 누적
        T_total = T @ T_total
        
        # src 갱신
        src_hom = np.hstack((src, np.ones((src.shape[0], 1), dtype=np.float32)))
        src_transformed = (T @ src_hom.T).T
        src = src_transformed[:, :2]
        
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
        
    return T_total, src, prev_error


def warp_bev_to_map(bev_image, T_total, scale):
    D = np.diag([scale, scale, 1])
    D_inv = np.linalg.inv(D)
    
    T_pixel = D_inv @ T_total @ D
    T_affine = T_pixel[:2, :]
    
    h, w = bev_image.shape[:2]
    warped_bev = cv2.warpAffine(bev_image, T_affine, (w, h),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped_bev

if __name__ == '__main__':
    # 파일 경로 (적절히 수정)
    bev_image_path = './bev_image1.png'  # BEV로 추출한 차선 이미지
    map_image_path = './zoom_img.png'  # 원본 맵 사진
    
    # 이미지 로드
    bev_image = cv2.imread(bev_image_path)
    map_image = cv2.imread(map_image_path)
    if bev_image is None or map_image is None:
        raise ValueError("이미지를 불러올 수 없습니다.")
    
    # 점군 추출
    bev_points = extract_points_from_image(bev_image_path)
    map_points = extract_points_from_image(map_image_path)
    
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
    