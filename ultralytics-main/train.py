from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    # ✅ 使用预训练模型
    model = YOLO("yolov8n.pt")

    # ✅ 加载你自己的 tennis.yaml
    model.train(data=r"D:\github\ultralytics-main\datasets\tennis\txt\tennisball.yaml", epochs=250)

if __name__ == '__main__':
    freeze_support()
    main()