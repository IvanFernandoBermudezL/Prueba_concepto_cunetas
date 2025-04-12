from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8 model
model = YOLO("/root/Yolo11_trainSegmentation_vast/yolo11n-seg.pt")

def main():
    # Train 
    model.train(
    data="/root/Yolo11_trainSegmentation_vast/data.yaml", 
    epochs=120, 
    batch=8, 
    imgsz=640, 
    save_period=5)

if __name__ == '__main__':
    main()