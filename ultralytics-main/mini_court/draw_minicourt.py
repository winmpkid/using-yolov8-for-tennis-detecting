import cv2
import numpy as np
from ultralytics import YOLO

def draw_tennis_court(width=240, height=480, ball_pos=None):
    court = np.full((height, width, 3), [200, 70, 10], dtype=np.uint8)
    line_color = (225, 225, 225)
    thickness = 4

    def draw_line(p1, p2):
        cv2.line(court, p1, p2, line_color, thickness)

    # 基本边界线
    draw_line((0, 0), (width - 1, 0))
    draw_line((width - 1, 0), (width - 1, height - 1))
    draw_line((width - 1, height - 1), (0, height - 1))
    draw_line((0, height - 1), (0, 0))

    # 单打边线
    single_left = int(width * 1.375 / 10.97)
    single_right = int(width * 9.595 / 10.97)
    draw_line((single_left, 0), (single_left, height - 1))
    draw_line((single_right, 0), (single_right, height - 1))

    # 中线
    cv2.line(court, (0, height // 2), (width, height // 2), line_color, thickness)

    # 发球线
    sline_top = int(height * (0.5 - 6.4 / 23.77))
    sline_bot = int(height * (0.5 + 6.4 / 23.77))
    draw_line((single_left, sline_top), (single_right, sline_top))
    draw_line((single_left, sline_bot), (single_right, sline_bot))

    # 中心 T 线
    draw_line((width // 2, sline_top), (width // 2, sline_bot))

    # 小球绘制
    if ball_pos is not None:
        bx, by = int(ball_pos[0]), int(ball_pos[1])
        if 0 <= bx < width and 0 <= by < height:
            cv2.circle(court, (bx, by), 10, (0, 0, 255), -1)
            cv2.circle(court, (bx, by), 12, (0, 0, 200), 2)
    return court

def main():
    model = YOLO('runs/detect/train2/weights/best.pt')
    video_path = r"D:\tennis\vedio\屏幕录制 2025-07-02 115316.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ 无法打开视频")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = r'D:\tennis\vedio\output_with_minicourt.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 你上传截图对应的标定点（按球场四角）
    src_pts = np.array([
        [826, 576], [2042, 574],[2456, 1382] , [408, 1378]     # 左下
    ], dtype=np.float32)

    dst_pts = np.array([
        [0, 0],
        [240, 0],
        [240, 480],
        [0, 480]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)

    cv2.namedWindow("Tennis Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tennis Detection", 960, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.20)[0]
        annotated = results.plot()
        ball_pos_pixel = None
        max_conf = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            if class_name.lower() == "tennisball" and conf > max_conf:
                max_conf = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                pt = np.array([[[cx, cy]]], dtype=np.float32)
                projected = cv2.perspectiveTransform(pt, H)[0][0]
                ball_pos_pixel = projected

        mini_court = draw_tennis_court(240, 480, ball_pos_pixel)

        # 合成俯视图
        x_offset = width - 240 - 20
        y_offset = 20

        if y_offset >= 0 and x_offset >= 0:
            annotated[y_offset:y_offset+480, x_offset:x_offset+240] = mini_court
            if ball_pos_pixel is not None:
                cv2.putText(annotated, f"Ball @ ({int(ball_pos_pixel[0])},{int(ball_pos_pixel[1])})",
                            (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out.write(annotated)
        cv2.imshow("Tennis Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
