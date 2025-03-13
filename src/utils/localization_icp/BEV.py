# 이미지 bev 만드는 코드
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 원본 이미지 로드
def convert_bev(img):
    # img = cv2.imread(img)
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img
    h, w = image.shape[:2]

    # 차선의 시작점과 끝점을 수동으로 설정 (예제 값)
    left_line_start = (int(25), int(h))  # 왼쪽 차선 시작점
    left_line_end = (int(135), int(140))    # 왼쪽 차선 끝점
    right_line_start = (int(445), int(h)) # 오른쪽 차선 시작점
    right_line_end = (int(340), int(140))   # 오른쪽 차선 끝점

    # 원본 이미지에서 차선이 위치한 네 개의 좌표 설정
    src_pts = np.float32([
        left_line_start,
        left_line_end,
        right_line_start,
        right_line_end
    ])

    # 변환 후, 차선을 평행하게 만드는 목표 좌표
    dst_pts = np.float32([
        [int(0.35*w), h],  # 왼쪽 차선 하단
        [int(0.35*w), h * 0.5],  # 왼쪽 차선 상단
        [int(0.65*w), h],  # 오른쪽 차선 하단
        [int(0.65*w), h * 0.5]   # 오른쪽 차선 상단
    ])
    def get_line_eq(p1, p2):
        """ 주어진 두 점을 지나는 직선의 기울기(m)와 y절편(b) 계산 """
        x1, y1 = p1
        x2, y2 = p2
        m = (y2 - y1) / (x2 - x1 + 1e-6)
        b = y1 - m * x1  
        return m, b

    m_left, b_left = get_line_eq(left_line_start, left_line_end)
    m_right, b_right = get_line_eq(right_line_start, right_line_end)
    x_min = 0
    x_max = w - 1
    y_min_left = int(m_left * x_min + b_left)
    y_max_left = int(m_left * x_max + b_left)
    y_min_right = int(m_right * x_min + b_right)
    y_max_right = int(m_right * x_max + b_right)

    image_with_lines = image.copy()
    cv2.line(image_with_lines, (x_min, y_min_left), (x_max, y_max_left), (255, 0, 0), 3)  # 왼쪽 차선 (파란색)
    cv2.line(image_with_lines, (x_min, y_min_right), (x_max, y_max_right), (0, 255, 0), 3) # 오른쪽 차선 (초록색)


    # Homography 행렬 자동 계산
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 변환 적용 (BEV)
    bev_image = cv2.warpPerspective(image, H, (w, h))
    # cv2.imwrite("bev_image1.png",bev_image)
    
    # 결과 출력
    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1)
    # plt.title("Original Image")
    # plt.imshow(image_with_lines)
    # plt.scatter(*zip(*src_pts), color='red', marker='o')  # 차선 위치 표시

    # plt.subplot(1,2,2)
    # plt.title("BEV (Perspective Transformed)")
    # plt.imshow(bev_image)
    # plt.scatter(*zip(*dst_pts), color='blue', marker='o')  # 변환된 좌표 표시

    # plt.show()

    return bev_image

if __name__ == '__main__':
    image = cv2.imread("./test_img/mask1.jpg")
    cv2.imshow("original", image)