from ultralytics import YOLO
import cv2

# ✅ 使用你自己训练出来的模型（不是 yolov8n.pt，而是 best.pt）
model = YOLO('runs/detect/train/weights/best.pt')

# 加载图片
img_path = r"D:\ultralytics-main\datasets\tennis\images\train\屏幕截图 2025-07-02 154143.png"

img = cv2.imread(img_path)

# 使用模型进行推理
results = model(img)

# 显示预测结果
results[0].show()

# 可选：保存结果图片
results[0].save(filename='result.jpg')
