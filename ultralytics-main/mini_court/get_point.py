import cv2
import numpy as np

clicked_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"第 {len(clicked_points)} 个点：({x}, {y})")

# 从视频读取第一帧
video_path = r"D:\tennis\vedio\屏幕录制 2025-07-01 103052.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ 无法读取视频帧")
    exit()

# 获取屏幕分辨率
screen_width = 1920  # 默认值，如果获取失败使用这个
screen_height = 1080
try:
    screen_width = cv2.getWindowImageRect('dummy')[2]
    screen_height = cv2.getWindowImageRect('dummy')[3]
except:
    pass

# 计算适合屏幕的窗口大小
frame_height, frame_width = frame.shape[:2]
scale = min(0.8 * screen_width / frame_width, 0.8 * screen_height / frame_height)
window_width = int(frame_width * scale)
window_height = int(frame_height * scale)

# 显示图像并绑定鼠标
cv2.namedWindow("点击选择球场四个角点", cv2.WINDOW_NORMAL)
cv2.resizeWindow("点击选择球场四个角点", window_width, window_height)
cv2.setMouseCallback("点击选择球场四个角点", mouse_callback)

while True:
    temp_frame = frame.copy()
    for pt in clicked_points:
        cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)

    cv2.imshow("点击选择球场四个角点", temp_frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(clicked_points) == 4:
        break

cv2.destroyAllWindows()

if len(clicked_points) == 4:
    print("\n✅ 你选择的 4 个角点如下（顺序要和目标小地图一致）：")
    for i, pt in enumerate(clicked_points):
        print(f"src_pts[{i}] = {pt}")
else:
    print("❌ 点的数量不足 4 个，请重新运行。")