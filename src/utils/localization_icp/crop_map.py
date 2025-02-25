import cv2
import numpy as np
import math

def extract_rotated_roi(image, rotated_pts, zoom_width, zoom_height):
    # 목적 좌표 (변환 후 크기)
    dst_pts = np.array([
        [0, 0],
        [zoom_width - 1, 0],
        [zoom_width - 1, zoom_height - 1],
        [0, zoom_height - 1]
    ], dtype=np.float32)

    # 투시 변환 행렬 계산
    M = cv2.getPerspectiveTransform(rotated_pts.astype(np.float32), dst_pts)

    # 변환 적용 (회전된 ROI 추출)
    roi = cv2.warpPerspective(image, M, (zoom_width, zoom_height))

    return roi


def zoom_in_on_region(image_path, x, y, heading, zoom_width=160, zoom_height=90):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 불러올 수 없습니다.")
        return
    
    h, w, _ = image.shape  # 이미지 크기
    
    # 기준점
    pivot_x, pivot_y = x, y + zoom_height // 2

    # ROI 좌표
    rect_pts = np.array([
        [x - zoom_width // 2, y - zoom_height],  # 좌상단
        [x + zoom_width // 2, y - zoom_height],  # 우상단
        [x + zoom_width // 2, y],  # 우하단
        [x - zoom_width // 2, y]   # 좌하단
    ], dtype=np.float32)

    # 회전 변환 행렬 생성
    M = cv2.getRotationMatrix2D((x, y), heading, 1.0)

    # 변환 행렬을 이용해 점 회전
    rotated_pts = cv2.transform(np.array([rect_pts]), M)[0].astype(int)

    # 회전된 좌표에서 최소 & 최대 x, y 값 찾기 (바운딩 박스)
    x1, y1 = np.min(rotated_pts, axis=0)  # 좌상단 좌표
    x2, y2 = np.max(rotated_pts, axis=0)  # 우하단 좌표
    
    # 영역이 이미지 범위를 벗어나면 조정
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    
    # 관심영역 추출
    return_width, return_height = 480, 270
    roi = extract_rotated_roi(image, rotated_pts, zoom_width, zoom_height)
    zoomed = np.zeros((return_height, return_width, 3), dtype=np.uint8)  # 검은색 배경
    
    # 확대할 영역이 있으면 확대 적용
    if roi.size > 0:
        resized_roi = cv2.resize(roi, (return_width, return_height), interpolation=cv2.INTER_CUBIC)
        zoomed[:resized_roi.shape[0], :resized_roi.shape[1]] = resized_roi
    
    # 원본 이미지에 표시 추가
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # 초기 위치 초록색 점
    
    # Rotated ROI 영역을 선으로 그림 (네 개 점을 이용)
    cv2.polylines(image, [rotated_pts], isClosed=True, color=(0, 0, 255), thickness=2)
    
    # 결과 출력
    # cv2.imshow("Original Image", image)
    # cv2.imshow("Zoomed Image", zoomed)
    # # cv2.imwrite("./zoom_img.png", zoomed)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return zoomed

# 예제 실행 (파일 경로와 확대 중심 좌표, 헤딩값 입력)
if __name__ == '__main__':
    image_path = "./test_img/SEAME_map.png"
    zoom_in_on_region(image_path, x=450, y=200, heading=45)
