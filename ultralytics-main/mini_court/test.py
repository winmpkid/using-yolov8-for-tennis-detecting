import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import deque

TENNIS_COURT_WIDTH_M = 8.23
TENNIS_COURT_LENGTH_M = 23.77
ball_positions = deque(maxlen=5)


pixels_per_meter_width = 240 / TENNIS_COURT_WIDTH_M
pixels_per_meter_height = 480 / TENNIS_COURT_LENGTH_M

def draw_tennis_court(width=240, height=480, ball_pos=None, player_pos=None, ball_speed=None):
    court = np.full((height, width, 3), [200, 70, 10], dtype=np.uint8)
    line_color = (225, 225, 225)
    thickness = 4

    def draw_line(p1, p2):
        cv2.line(court, p1, p2, line_color, thickness)

    draw_line((0, 0), (width - 1, 0))
    draw_line((width - 1, 0), (width - 1, height - 1))
    draw_line((width - 1, height - 1), (0, height - 1))
    draw_line((0, height - 1), (0, 0))

    single_left = int(width * 1.375 / 10.97)
    single_right = int(width * 9.595 / 10.97)
    draw_line((single_left, 0), (single_left, height - 1))
    draw_line((single_right, 0), (single_right, height - 1))

    cv2.line(court, (0, height // 2), (width, height // 2), line_color, thickness)

    sline_top = int(height * (0.5 - 6.4 / 23.77))
    sline_bot = int(height * (0.5 + 6.4 / 23.77))
    draw_line((single_left, sline_top), (single_right, sline_top))
    draw_line((single_left, sline_bot), (single_right, sline_bot))

    draw_line((width // 2, sline_top), (width // 2, sline_bot))

    if ball_pos is not None:
        bx, by = int(ball_pos[0]), int(ball_pos[1])
        if 0 <= bx < width and 0 <= by < height:
            cv2.circle(court, (bx, by), 10, (0, 0, 255), -1)
            cv2.circle(court, (bx, by), 12, (0, 0, 200), 2)
            if ball_speed is not None:
                speed_text = f"{ball_speed:.1f} px/s"
                cv2.putText(court, speed_text, (bx + 15, by - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if player_pos is not None:
        bx, by = int(player_pos[0]), int(player_pos[1])
        if 0 <= bx <= width and 0 <= by <= height:
            cv2.circle(court, (bx, by), 10, (0, 255, 0), -1)
            cv2.circle(court, (bx, by), 12, (0, 200, 0), 2)

    return court

def pixel_to_meter(ball_pos_pixel):
    if ball_pos_pixel is None:
        return None
    x_meters = ball_pos_pixel[0] / pixels_per_meter_width
    y_meters = ball_pos_pixel[1] / pixels_per_meter_height
    return (x_meters, y_meters)

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
    frame_time = 1.0 / fps

    output_path = r'D:\tennis\vedio\output_with_minicourt.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    src_pts = np.array([[826, 576], [2042, 574],[2456, 1382] , [408, 1378]], dtype=np.float32)
    dst_pts = np.array([[0, 0],[240, 0],[240, 480],[0, 480]], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)

    cv2.namedWindow("Tennis Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tennis Detection", 960, 720)

    prev_ball_pos = None
    ball_speed_mps = None
    max_ball_speed = -1
    current_ball_speed = -1
    time_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.20)[0]
        annotated = frame.copy()

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if model.names[int(box.cls[0])] == "tennisball" and ball_speed_mps is not None:
                cv2.putText(annotated, f"{ball_speed_mps:.1f} m/s", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if model.names[int(box.cls[0])] == "tennisplayer":
                cv2.putText(annotated, "tennisplayer", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 150), 2)

        ball_pos_pixel = None
        max_conf = 0
        point = None
        time_fps+=1

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            if class_name.lower() == "tennisball" and conf > max_conf:
                max_conf = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                point = np.array([cx, cy], dtype=np.float32)
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                projected = cv2.perspectiveTransform(pt, H)[0][0]
                ball_pos_pixel = projected
                if point is not None:
                    ball_positions.append(point)
                    time_fps = 0

            if time_fps >= 3:
                ball_positions.clear()

            if class_name.lower() == "tennisplayer":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, y2
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                player_pos_pixel = cv2.perspectiveTransform(pt, H)[0][0]
        
        if len(ball_positions) >= 1:
            for i, pos in enumerate(ball_positions):
                bx, by = int(pos[0]), int(pos[1])
                alpha = 255 - i * 40
                cv2.circle(annotated, (bx, by), 8, (0, alpha, 0), -1)

        if ball_pos_pixel is not None and prev_ball_pos is not None:
            prev_pos_meters = pixel_to_meter(prev_ball_pos)
            curr_pos_meters = pixel_to_meter(ball_pos_pixel)
            dx = curr_pos_meters[0] - prev_pos_meters[0]
            dy = curr_pos_meters[1] - prev_pos_meters[1]
            distance_meters = math.sqrt(dx**2 + dy**2)
            ball_speed_mps = distance_meters / frame_time

            if ball_speed_mps > 70:
                ball_speed_mps = 0
            current_ball_speed = ball_speed_mps

            if int(ball_speed_mps) > max_ball_speed:
                max_ball_speed = ball_speed_mps

            # ✅ 击球检测：y轴突变则清空残影
            dy_pixel = abs(ball_pos_pixel[1] - prev_ball_pos[1])
            if dy_pixel > 40:
                ball_positions.clear()
                cv2.putText(annotated, "Hit detected!", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            ball_speed_mps = None

        if max_ball_speed is not None:
            cv2.putText(annotated, f"max_Speed: {max_ball_speed:.1f} m/s", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        prev_ball_pos = ball_pos_pixel if ball_pos_pixel is not None else prev_ball_pos
        mini_court = draw_tennis_court(240, 480, ball_pos_pixel, player_pos_pixel, ball_speed_mps)

        x_offset = width - 240 - 20
        y_offset = 20
        cv2.putText(annotated, f"current_Speed: {current_ball_speed:.1f} m/s", 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if y_offset >= 0 and x_offset >= 0:
            annotated[y_offset:y_offset+480, x_offset:x_offset+240] = mini_court
            if ball_pos_pixel is not None:
                cv2.putText(annotated, f"Ball @ ({int(ball_pos_pixel[0])},{int(ball_pos_pixel[1])})",
                            (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if ball_speed_mps is not None:
                    cv2.putText(annotated, f"Speed: {ball_speed_mps:.1f} m/s", 
                                (x_offset, y_offset + 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(annotated)
        cv2.imshow("Tennis Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
