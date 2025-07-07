from ultralytics import YOLO
import cv2

def main():
    model = YOLO('runs/detect/train/weights/best.pt')  # 替换成你的模型路径

    video_path = r"D:\tennis\vedio\屏幕录制 2025-07-02 115316.mp4" # 输入视频路径
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频的宽高和帧率，用于VideoWriter
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 定义输出视频路径和编码器，注意输出文件格式支持的编码
    output_path = r'D:\tennis\vedio\result3_video.mp4'  # 输出视频路径

    # 推荐用mp4v编码，Windows下常用
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cv2.namedWindow('YOLOv8 Video Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLOv8 Video Detection', 640, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        # 写入帧到输出视频
        out.write(annotated_frame)

        cv2.imshow('YOLOv8 Video Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
