import cv2
from ultralytics import YOLO
import random

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
MODEL_PATH = "C:/wajahat/hand_in_pocket/detection_models/train/exp_3/weights/best.pt"        # your custom model
VIDEO_PATH = "C:/wajahat/hand_in_pocket/new_test_bench3/r1_d1_fp1.mp4"      # input video
# OUTPUT_PATH = "output.mp4"    # output file
CONF_THRESHOLD = 0.25         # prediction threshold
# -----------------------------------------------------------


# Load model
model = YOLO(MODEL_PATH)

# Generate a unique color for each class
# model.names = {0:'classA', 1:'classB', ...}
class_names = model.names
num_classes = len(class_names)

# Assign random colors to each class
colors = {}
for cls_id in range(num_classes):
    colors[cls_id] = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )

# Read input video
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define video writer
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))


print("Running inference...")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    # Run inference
    results = model(frame, conf=CONF_THRESHOLD)

    frame = cv2.resize(frame,(1280,720))
    # YOLO outputs a list per frame, so take results[0]
    dets = results[0].boxes

    if dets is not None:
        for box in dets:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            x1, y1, x2, y2 = map(int, xyxy)

            # Get label + color
            label = f"{class_names[cls_id]} {conf:.2f}"
            color = colors[cls_id]

            # Draw rectangular bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label above box
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    # Save frame
    # out.write(frame)
    cv2.imshow("frame",frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
# out.release()
print("\n✅ Inference complete!")
# print(f"Saved output video: {OUTPUT_PATH}")
