import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import json
import csv
from collections import deque
from ultralytics import YOLO
import statistics as stats
import random

# ========== MLP Model ==========
class MLP(nn.Module):
    def __init__(self, input_size=104, hidden_size=64):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(64, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
# ========== CNN Block ===============
def draw_cnn_boxes(frame, roi_data_list):
    for roi in roi_data_list:
        x1, y1, x2, y2 = roi["xmin"], roi["ymin"], roi["xmax"], roi["ymax"]
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        results = cnn_model(crop, conf=cnn_conf, verbose=False)
        boxes = results[0].boxes

        if boxes is None:
            continue

        CNN_COLORS = {}
        for cls_id, cls_name in cnn_model.names.items():
            random.seed(cls_id)
            CNN_COLORS[cls_name] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
        )

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = cnn_model.names[cls_id]

            bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
            fx1, fy1 = x1 + bx1, y1 + by1
            fx2, fy2 = x1 + bx2, y1 + by2

            color = CNN_COLORS.get(cls_name, (255, 255, 255))

            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
            cv2.putText(
                frame,
                f"{cls_name} {conf:.2f}",
                (fx1, fy1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )


def cnn_validate_action(frame, roi):
    """
    Returns True if alert is ALLOWED
    Returns False if alert must be SUPPRESSED
    """
    x1, y1, x2, y2 = roi["xmin"], roi["ymin"], roi["xmax"], roi["ymax"]
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        print("DEBUG: Crop size is 0, suppressing alert")
        return False  # fail-safe: suppress alert

    results = cnn_model(crop, conf=cnn_conf, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("DEBUG: No boxes detected, allowing alert")
        return True  # nothing detected → alert allowed
    
    CNN_COLORS = {}
    for cls_id, cls_name in cnn_model.names.items():
        random.seed(cls_id)
        CNN_COLORS[cls_name] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )

    print(f"DEBUG: Detected classes: {[cnn_model.names[int(box.cls[0])] for box in boxes]}")
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = cnn_model.names[cls_id]
        conf = float(box.conf[0])
        print(f"DEBUG: Class '{cls_name}' detected with conf {conf}")

        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())

        # Convert to full-frame coordinates
        fx1, fy1 = x1 + bx1, y1 + by1
        fx2, fy2 = x1 + bx2, y1 + by2

        color = CNN_COLORS.get(cls_name, (255, 255, 255))

        # Draw box
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)

        # Draw label
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (fx1, fy1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        if cls_name in CNN_BLOCK_CLASSES:
            print(f"DEBUG: Blocking class '{cls_name}' detected, suppressing alert")
            return False  # suppress alert

    print("DEBUG: No blocking classes detected, allowing alert")
    return True  # no blocking class found


# ========== ROI Assignment ==========
roi_data_list = []
def assign_roi_index(x):
    for roi in roi_data_list:
        if roi["xmin"] <= x < roi["xmax"]:
            return roi["position"]

# ========== Load Model ==========
def load_mlp_model(weights_path, device):
    model = MLP()
    # model.load_state_dict(torch.load(weights_path, map_location=device)) # for without schedular trained models
    checkpoint = torch.load(weights_path)  # map_location ensures compatibility with CPU/GPU
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(device)
    model.eval()
    return model

mlp_times = []
video_num = 0

# ========== Main Inference ==========
if __name__ == "__main__":
    kp_model = YOLO("C:/wajahat/hand_in_pocket/bestv8-1.pt", verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # mlp_model_name = "mlp_temp_norm_regrouped_pos_gen_augmented_round-c0"
    # mlp_model_name = "mlp_temp_regrouped_pos_gen_round-c0-moiz-t3"
    mlp_model_name = "mlp_temp_balanced_norm-wajahat-t3-c0"
    mlp_model = load_mlp_model(f"rf_models/{mlp_model_name}.pt", device)
    cnn_model = YOLO("C:/wajahat/hand_in_pocket/detection_models/train/t2(5_classes)/weights/best.pt", verbose=False)
    cnn_conf = 0.25
    input_dir = "C:/wajahat/hand_in_pocket/new_test_bench3"
    # input_dir = "F:/Wajahat/qiyas_analysis/aug_5-2/Hands In Pocket/TP"
    json_path = "qiyas_multicam.camera_final.json"  # system 1 json
    # json_path = "qiyas_multicam_2.camera.json"    # system 2 json
    WINDOW_SIZE = 3
    waitkey = 2
    SKIP_RATE = 1
    ALERT_THRESHOLD = 5
    frame_idx = 0
    prediction_streak = {}
    # camera_id = "camera_5"
    user_input = True

    CNN_BLOCK_CLASSES = {
        # "Hand in pocket" , #
        # "Hand in pocket suspected" , #
        # "Pocket" , #
        "Hand on an arm rest" ,
        "Hand on desk" , #
        "Hand on lap" ,
        "chair hand rest" , #
        "Hand on head" ,
        "hand under desk" ,
        "hand on back" 
    }

    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4') or f.endswith('.avi')]
    if not video_files:
        print("No videos found.")
        exit()

    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        print(f"Processing: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error loading video.")
            continue

        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Select Camera", frame)
        cv2.waitKey(1)

        with open(json_path, 'r') as f:
            camera_config = json.load(f)

        skip_video = False
        # user_input = False
        while True:
            if user_input == True:
                cam_id = input("Enter camera ID: ")
                if cam_id.lower() == 's':
                    skip_video = True
                    cap.release()
                    cv2.destroyWindow("Select Camera")
                    break
                cam_key = f"camera_{cam_id}"
            else:
                # cam_key = camera_id
                user = input("Press Enter")
            camera_data = next((cam for cam in camera_config if cam["_id"] == cam_key), None)
            if camera_data:
                break
            print("Invalid camera ID. Try again.")

        if skip_video:
            continue

        cv2.destroyWindow("Select Camera")
        roi_data_list = list(camera_data["data"].values())
        roi_lookup = {roi["position"]: roi for roi in roi_data_list}
        sliding_window = {}

        frame_number = 0
        csv_output = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            frame_number += 1
            if frame_idx % SKIP_RATE != 0:
                continue

            frame = cv2.resize(frame, (1280, 720))
            results = kp_model(frame, verbose=False)
            current_detected = set()
            waitkey = 1
            for result in results:
                if not hasattr(result, 'keypoints') or result.keypoints is None:
                    continue

                keypoints_tensor = result.keypoints.data

                for person_idx, kp_tensor in enumerate(keypoints_tensor):
                    keypoints = []
                    feature_dict = {}

                    # print(f"kp tensor: {kp_tensor}")

                    for i, keypoint in enumerate(kp_tensor):
                        # waitkey = 100
                        x, y, conf = keypoint[:3].cpu().numpy()
                        x = x.astype(int)
                        y = y.astype(int)
                        # print(f"frame: {frame_number}, x: {x}, y: {y}, conf: {conf}")
                        if conf > 0.5:
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                        # for normalized keypoints 
                        if conf < 0.5:
                            x, y = -1, -1
                        else:
                            x = (x / 1280).round(3)
                            y = (y / 720).round(3)
                        feature_dict[f"kp_{i}_x"] = x
                        feature_dict[f"kp_{i}_y"] = y
                        keypoints.append((x, y))
                        # print(f" keypoints: {keypoints}")

                    if len(keypoints) == 0 or all((x == -1 and y == -1) for x, y in keypoints):
                        continue

                    person_x = keypoints[0][0] * 1280

                    position = assign_roi_index(person_x)
                    roi_data = roi_lookup.get(position)
                    if not roi_data:
                        continue

                    feature_dict['position'] = position
                    if position not in sliding_window:
                        sliding_window[position] = deque(maxlen=WINDOW_SIZE)
                    sliding_window[position].append(feature_dict)

                    if len(sliding_window[position]) == WINDOW_SIZE:
                        flat_feature = {}
                        for i in range(10):
                            for axis in ['x', 'y']:
                                for t in range(WINDOW_SIZE):
                                    flat_feature[f"kp_{i}_{axis}_t{t}"] = sliding_window[position][t][f"kp_{i}_{axis}"]
                        pos_list = roi_data.get("position_list", [0, 0, 0, 0])
                        flat_feature["position_a"] = pos_list[0]
                        flat_feature["position_b"] = pos_list[1]
                        flat_feature["position_c"] = pos_list[2]
                        flat_feature["position_d"] = pos_list[3]

                        ordered_columns = [f"kp_{i}_{axis}_t{t}" for i in range(10) for axis in ['x', 'y'] for t in range(WINDOW_SIZE)]
                        ordered_columns.extend(["position_a", "position_b", "position_c", "position_d"])

                        input_tensor = torch.tensor([[flat_feature[col] for col in ordered_columns]], dtype=torch.float32).to(device)
                        with torch.no_grad():
                            prob = mlp_model(input_tensor).item()
                            prediction = 1 if prob >= 0.5 else 0

                        desk_id = roi_data['desk']
                        streak_key = (desk_id, person_idx)

                        current_detected.add(streak_key)
                        if prediction == 1:
                            # print(f"hand in pocket")
                            prediction_streak[streak_key] = prediction_streak.get(streak_key, 0) + 1
                        else:
                            prediction_streak[streak_key] = 0

                        # draw_cnn_boxes(frame, roi_data_list)

                        # waitkey = 0 if prediction == 1 else 1

                        if prediction_streak.get(streak_key, 0) >= ALERT_THRESHOLD:
                            # alert_label = f"ALERT: {desk_id} - Person {person_idx}"
                            waitkey = 60
                            alert_label = f"ALERT: {desk_id} - Hand in Pocket - MLP"
                            cv2.putText(frame, alert_label, (int(person_x), 150 + person_idx * 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            allow_alert = cnn_validate_action(frame, roi_data)

                            if allow_alert:
                                alert_label = f"ALERT: {desk_id} - Hand in Pocket - CNN"
                                cv2.putText(frame, alert_label, (int(person_x), 250 + person_idx * 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            print(f"************ALERT - Desk:{desk_id}**************")
                        # waitkey = 1

                        label = "Hand in Pocket" if prediction else "No Hand in Pocket"
                        # label = "No Hand in Pocket"
                        # color = (0, 255, 0)
                        color = (0, 0, 255) if prediction else (0, 255, 0)
                        cv2.putText(frame, f"{label} ({prob:.2f})", (int(person_x), 50 + person_idx * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"Desk: {desk_id}, Pos: {position}",
                                    (int(person_x), 100 + person_idx * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 180, 0), 2)

                        # Save prediction to CSV row

                        row = {
                            "frame": frame_number,
                            "desk": desk_id,
                            # "person_idx": person_idx,
                            "prediction": prediction,
                            "probability": round(prob, 4)
                        }

                        for t in range(WINDOW_SIZE):
                            for i in range(10):
                                row[f"kp_{i}_x_t{t}"] = sliding_window[position][-1][f"kp_{i}_x"]
                                row[f"kp_{i}_y_t{t}"] = sliding_window[position][-1][f"kp_{i}_y"]

                        csv_output.append(row)

            for key in list(prediction_streak.keys()):
                if key not in current_detected:
                    prediction_streak[key] = 0

            cv2.imshow("MLP Inference", frame)
            # if cv2.waitKey(waitkey) & 0xFF == ord('q'):
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        video_num += 1
        print(f"video: {video_num}")
        cap.release()

        # Save CSV file
        csv_name = os.path.splitext(video_file)[0] + ".csv"
        output_folder = f"C:/wajahat/hand_in_pocket/dataset/results_csv/{mlp_model_name}0"
        os.makedirs(output_folder, exist_ok=True)
        csv_path = os.path.join(output_folder, csv_name)
        with open(csv_path, 'w', newline='') as f:
            keypoint_cols = [f"kp_{i}_{axis}_t{t}" for i in range(10) for axis in ['x','y'] for t in range(WINDOW_SIZE)]
            fieldnames = ['frame', 'desk'] + keypoint_cols + ["prediction", "probability"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_output)

    cv2.destroyAllWindows()
