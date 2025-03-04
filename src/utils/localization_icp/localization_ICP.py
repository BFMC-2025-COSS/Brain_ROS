import numpy as np
import cv2
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import cupy as cp

def extract_points_from_image(image_path, threshold_val=127):
    # image = cv2.imread(image_path)
    #image = image_path
    image = cv2.pyrDown(image_path)
    image = cv2.pyrDown(image)

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
    src = cp.asarray(src)
    dst = cp.asarray(dst)

    centroid_src = cp.mean(src, axis=0)
    centroid_dst = cp.mean(dst, axis=0)
    # centroid_src = np.mean(src, axis=0)
    # centroid_dst = np.mean(dst, axis=0)
    
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst
    
    W = cp.dot(dst_centered.T, src_centered)
    U, _, Vt = np.linalg.svd(W)
    R = cp.dot(U, Vt)
    # W = np.dot(dst_centered.T, src_centered)
    # U, _, Vt = np.linalg.svd(W)
    # R = np.dot(U, Vt)

    # 반사행렬(det < 0) 보정
    # if np.linalg.det(R) < 0:
    #     Vt[-1, :] *= -1 
    #     R = np.dot(U, Vt)
    
    # t = centroid_dst - np.dot(R, centroid_src)
    if cp.linalg.det(R) < 0:
        Vt[-1, :] *= -1 
        R = cp.dot(U, Vt)
    
    t = centroid_dst - cp.dot(R, centroid_src)
    # return R, t
    return cp.asnumpy(R), cp.asnumpy(t)

def icp(src, dst, max_iterations=20, tolerance=1e-4):
    # src = source.copy()
    # dst = target.copy()

    T_total = np.eye(3, dtype=np.float32)
    prev_error = float('inf')
    
    for i in range(max_iterations):
        print(f"Iteration: {i}")
        distances, indices = nearest_neighbor(src, dst)
        matched_dst = dst[indices]
        print("Ready compute")
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
    T_total, aligned_bev_points, final_error = icp(map_points_phys, bev_points_phys)
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
    