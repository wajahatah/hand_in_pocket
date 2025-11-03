from ultralytics import YOLO
import os

# === CONFIGURATION ===
MODEL = "yolov8m.pt"        # choose from yolov8n/s/m/l/x.pt
DATA_YAML = "C:/wajahat/hand_in_pocket/dataset/images_bb/training1/data.yaml" #"C:\wajahat\hand_in_pocket\dataset\images_bb\training1\data.yaml"
EPOCHS = 100
IMG_SIZE = 640
# BATCH_SIZE = 8
DEVICE = 0                  # GPU id, or 'cpu'
PROJECT = "detection_models/train"
NAME = "exp_1"

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
        # augment=True,
        # mosaic=1.0,
        # mixup=0.1
    )

    print("\n✅ Training complete.")
    print(f"Results saved to: {os.path.join(PROJECT, NAME)}")
