from ultralytics import YOLO
import os

# === CONFIGURATION ===
MODEL = "yolov8m.pt"        # choose from yolov8n/s/m/l/x.pt
DATA_YAML = "C:/wajahat/hand_in_pocket/dataset/images_bb/training2/data.yaml" #"C:\wajahat\hand_in_pocket\dataset\images_bb\training1\data.yaml"
EPOCHS = 100
IMG_SIZE = 640
# BATCH_SIZE = 8
DEVICE = 0                  # GPU id, or 'cpu'
PROJECT = "detection_models/train"
NAME = "t2(2_classes)"

# === TRAINING ===
if __name__ == "__main__":
    model = YOLO(MODEL)  # load pretrained model

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        # batch=BATCH_SIZE,
        # device=DEVICE,
        project=PROJECT,
        name=NAME,
        # workers=8,
        patience=15,
        optimizer='AdamW',
        lr0=0.01,
        weight_decay=0.0005,
        # classes = [0,1,6,8,9,10] #without 5 classes approach
        # classes = [0,1,2,3,5,6,7,8,9,10] #without hand on desk class
        classes = [2,3] #only hip and suspected hip
        # augment=True,
        # mosaic=1.0,
        # mixup=0.1
    )

    print("\n✅ Training complete.")
    print(f"Results saved to: {os.path.join(PROJECT, NAME)}")
