import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import json
import csv
import math
import re
from collections import deque
from ultralytics import YOLO

# =========================
# ===== CONFIG / FLAGS ====
# =========================
INPUT_DIR = "F:/Wajahat/qiyas_analysis/aug_7/Hands In Pocket/TP"
ROI_JSON_PATH = "qiyas_multicam.camera_final.json"
YOLO_WEIGHTS = "bestv8-1.pt"
MLP_NAME = "mlp_temp_regrouped_pos_gen_ROI_norm-c0"  # weights in rf_models/{MLP_NAME}.pt
MLP_WEIGHTS = f"rf_models/{MLP_NAME}.pt"
OUTPUT_DIR = f"C:/wajahat/hand_in_pocket/dataset/results_csv/{MLP_NAME}"

FRAME_RESIZE = (1280, 720)
WINDOW_SIZE = 5
SKIP_RATE = 1            # process every SKIP_RATEth frame
WAITKEY = 1              # viz delay
ALERT_THRESHOLD = 5
CONF_THR = 0.5           # keypoint conf threshold
VIDEO = 0

# ROI transforms
APPLY_ROI_SUBTRACTION = True       # x-=xmin, y-=ymin (original zeros remain 0)
DROP_NEG_AFTER_SUB = False         # drop sample if any kp<0 post-subtraction
APPLY_ROI_NORMALIZATION = True     # normalize using ROI
NORM_ASSUMES_SUBTRACTED = True     # if True, x/=width, y/=height; else (x-xmin)/width...
NORM_ZERO_TO_NEG1 = True           # convert original zeros to -1 on normalization
NORM_CLAMP_01 = False              # clamp normalized outputs to [0,1]

# ---- ROI overlay drawing ----
SHOW_ROI_OVERLAY   = False           # <--- toggle ON/OFF
ROI_ALPHA          = 0.75           # 0 = no blend; 0.25 = soft overlay
ROI_RECT_THICKNESS = 2
ROI_LINE_THICKNESS = 2
ROI_FONT_SCALE     = 0.6
ROI_FONT_THICK     = 2
# video_num = 0

# =========================
# ======= MODELS ==========
# =========================
class MLP(nn.Module):
    def __init__(self, input_size=104, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

def load_mlp_model(weights_path, device):
    model = MLP(input_size=104, hidden_size=64)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# =========================
# ===== ROI HELPERS =======
# =========================
def load_camera_config(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_roi_lists(camera_data):
    """Return (roi_list, roi_lookup_by_position) for a single camera node."""
    roi_list = list(camera_data["data"].values())
    roi_lookup = {roi["position"]: roi for roi in roi_list}
    return roi_list, roi_lookup

def assign_roi_index(x, roi_list):
    """Given x, find roi['position'] where xmin <= x < xmax."""
    for roi in roi_list:
        if roi["xmin"] <= x < roi["xmax"]:
            return roi["position"]
    return None

# --------- ROI overlay drawing ----------
def _color_for_position(pos: int):
    # Stable distinct BGR colors for small pos values
    palette = [
        (36, 255, 12),   # green-ish
        (0, 165, 255),   # orange
        (255, 0, 0),     # blue
        (147, 20, 255),  # purple
        (0, 255, 255),   # yellow
        (255, 255, 0),   # cyan
    ]
    return palette[pos % len(palette)]

def draw_roi_overlay(frame_bgr, roi_list, alpha=ROI_ALPHA, draw_baselines=True):
    """
    Draw ROI boxes + labels (and baselines if present) on a copy of frame.
    Uses alpha blend if 0 < alpha <= 1.
    """
    if not roi_list:
        return frame_bgr

    base = frame_bgr.copy()
    canvas = frame_bgr.copy()

    for roi in roi_list:
        xmin = int(roi["xmin"]); ymin = int(roi["ymin"])
        xmax = int(roi["xmax"]); ymax = int(roi["ymax"])
        pos  = int(roi.get("position", -1))
        desk = roi.get("desk", "?")
        color = _color_for_position(pos)

        # Rectangle
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, ROI_RECT_THICKNESS)

        # Baseline (if keys exist)
        if draw_baselines and all(k in roi for k in ("left_x", "left_y", "right_x", "right_y")):
            p1 = (int(roi["left_x"]), int(roi["left_y"]))
            p2 = (int(roi["right_x"]), int(roi["right_y"]))
            cv2.line(canvas, p1, p2, color, ROI_LINE_THICKNESS)

        # Label (desk + position)
        label = f"Desk {desk} | Pos {pos}"
        cv2.putText(canvas, label, (xmin + 5, max(ymin - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, ROI_FONT_SCALE, color, ROI_FONT_THICK, cv2.LINE_AA)

    if alpha and 0.0 < alpha < 1.0:
        blended = cv2.addWeighted(canvas, alpha, base, 1 - alpha, 0)
        return blended
    else:
        return canvas

# --------- KP field iterators & transforms ----------
_KP_SINGLE_RE = re.compile(r"^kp_(\d+)_(x|y)$", re.IGNORECASE)

def _iter_kps(feature_dict):
    """Yield (key, axis, value) for keys like 'kp_0_x' and 'kp_0_y'."""
    for k, v in feature_dict.items():
        m = _KP_SINGLE_RE.match(k)
        if m:
            yield k, m.group(2).lower(), v

def roi_subtract_feature_dict(feature_dict, roi, keep_zero=True):
    """
    In-place subtraction: x -= xmin, y -= ymin.
    keep_zero=True preserves zeros as zeros (missing points).
    Returns True if any resulting kp becomes negative.
    """
    xmin, ymin = float(roi["xmin"]), float(roi["ymin"])
    any_negative = False
    for k, axis, val in _iter_kps(feature_dict):
        if keep_zero and val == 0:
            continue
        delta = xmin if axis == "x" else ymin
        new_val = float(val) - delta
        feature_dict[k] = new_val
        if new_val < 0:
            any_negative = True
    return any_negative

def roi_normalize_feature_dict(feature_dict, roi,
                               zero_to_neg1=True,
                               assume_subtracted=True,
                               clamp01=False):
    """
    In-place normalization using ROI.
      if assume_subtracted:   x/=width, y/=height
      else:                   x=(x-xmin)/width, y=(y-ymin)/height
    Zeros (missing) -> -1 when zero_to_neg1=True.
    width/height <= 0 -> -1
    """
    xmin, ymin = float(roi["xmin"]), float(roi["ymin"])
    width = float(roi["xmax"]) - float(roi["xmin"])
    height = float(roi["ymax"]) - float(roi["ymin"])

    def _safe_norm(val, axis):
        if zero_to_neg1 and val == 0:
            return -1.0
        num = float(val) if assume_subtracted else float(val) - (xmin if axis == "x" else ymin)
        den = width if axis == "x" else height
        if den <= 0 or math.isclose(den, 0.0):
            return -1.0
        out = num / den
        if clamp01:
            out = 0.0 if out < 0 else (1.0 if out > 1 else out)
        return out

    for k, axis, val in _iter_kps(feature_dict):
        feature_dict[k] = _safe_norm(val, axis)

# =========================
# ===== MAIN PIPELINE =====
# =========================
def main():
    # --- Load detectors/models ---
    kp_model = YOLO(YOLO_WEIGHTS, verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = load_mlp_model(MLP_WEIGHTS, device)

    # --- Gather videos ---
    video_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".mp4")]
    if not video_files:
        print("No videos found.")
        return

    camera_config = load_camera_config(ROI_JSON_PATH)
    frame_idx_global = 0
    prediction_streak = {}  # (desk_id, person_idx) -> streak length

    for video_file in video_files:
        video_path = os.path.join(INPUT_DIR, video_file)
        print(f"Processing: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error loading video.")
            continue

        # Show first frame to help pick camera
        ok, frame0 = cap.read()
        if not ok:
            cap.release()
            continue
        frame0 = cv2.resize(frame0, FRAME_RESIZE)
        cv2.imshow("Select Camera", frame0)
        cv2.waitKey(1)

        # Camera selection (interactive)
        while True:
            cam_id = input("Enter camera ID (or 's' to skip this video): ").strip()
            if cam_id.lower() == "s":
                cap.release()
                cv2.destroyWindow("Select Camera")
                break
            cam_key = f"camera_{cam_id}"
            camera_data = next((cam for cam in camera_config if cam["_id"] == cam_key), None)
            if camera_data:
                cv2.destroyWindow("Select Camera")
                break
            print("Invalid camera ID. Try again.")

        if not cap.isOpened():
            continue  # skipped

        roi_list, roi_lookup_by_position = build_roi_lists(camera_data)
        sliding_window = {}  # position -> deque of feature_dicts (len=WINDOW_SIZE)
        csv_output = []

        # Reset stream to beginning after showing first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_number = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx_global += 1
            frame_number += 1
            if frame_idx_global % SKIP_RATE != 0:
                continue

            # Inference frame
            frame = cv2.resize(frame, FRAME_RESIZE)
            results = kp_model(frame, verbose=False)

            # Visualization frame (optionally overlay ROIs)
            vis = frame.copy()
            if SHOW_ROI_OVERLAY:
                vis = draw_roi_overlay(vis, roi_list, alpha=ROI_ALPHA, draw_baselines=True)


            current_detected = set()

            WAITKEY = 1

            for result in results:
                if not hasattr(result, "keypoints") or result.keypoints is None:
                    continue

                kps_tensor = result.keypoints.data  # [num_persons, K, 3]
                for person_idx, kp_tensor in enumerate(kps_tensor):
                    keypoints = []
                    feature_dict = {}

                    # Extract 10 keypoints (x,y) with zero for low conf
                    for i, keypoint in enumerate(kp_tensor):
                        WAITKEY = 30
                        x, y, conf = keypoint[:3].cpu().numpy()
                        x_i = int(x)
                        y_i = int(y)
                        if conf > CONF_THR:
                            cv2.circle(vis, (x_i, y_i), 5, (0, 255, 0), -1)
                        if conf < CONF_THR:
                            x_i, y_i = 0, 0  # missing
                        feature_dict[f"kp_{i}_x"] = float(x_i)
                        feature_dict[f"kp_{i}_y"] = float(y_i)
                        keypoints.append((x_i, y_i))

                    if len(keypoints) == 0 or all((x == 0 and y == 0) for x, y in keypoints):
                        continue

                    # Assign ROI
                    person_x = keypoints[0][0]
                    position = assign_roi_index(person_x, roi_list)
                    if position is None:
                        continue
                    roi = roi_lookup_by_position.get(position)
                    if not roi:
                        continue

                    # --- ROI transforms ---
                    if APPLY_ROI_SUBTRACTION:
                        neg = roi_subtract_feature_dict(feature_dict, roi, keep_zero=True)
                        if DROP_NEG_AFTER_SUB and neg:
                            continue

                    if APPLY_ROI_NORMALIZATION:
                        roi_normalize_feature_dict(
                            feature_dict,
                            roi=roi,
                            zero_to_neg1=NORM_ZERO_TO_NEG1,
                            assume_subtracted=NORM_ASSUMES_SUBTRACTED,
                            clamp01=NORM_CLAMP_01,
                        )

                    # Maintain sliding window per position
                    if position not in sliding_window:
                        sliding_window[position] = deque(maxlen=WINDOW_SIZE)
                    sliding_window[position].append(feature_dict)

                    # Run MLP once window is full
                    if len(sliding_window[position]) == WINDOW_SIZE:
                        # Flatten window -> 10 kps * 2 axes * T + 4 positional
                        flat = {}
                        for i in range(10):
                            for axis in ("x", "y"):
                                for t in range(WINDOW_SIZE):
                                    flat[f"kp_{i}_{axis}_t{t}"] = sliding_window[position][t][f"kp_{i}_{axis}"]

                        pos_list = roi.get("position_list", [0, 0, 0, 0])
                        flat["position_a"], flat["position_b"], flat["position_c"], flat["position_d"] = (
                            float(pos_list[0]), float(pos_list[1]), float(pos_list[2]), float(pos_list[3])
                        )

                        ordered_cols = [f"kp_{i}_{axis}_t{t}"
                                        for i in range(10)
                                        for axis in ("x", "y")
                                        for t in range(WINDOW_SIZE)]
                        ordered_cols += ["position_a", "position_b", "position_c", "position_d"]

                        x_tensor = torch.tensor([[flat[col] for col in ordered_cols]], dtype=torch.float32, device=device)
                        with torch.no_grad():
                            prob = float(mlp(x_tensor).item())
                        pred = 1 if prob >= 0.5 else 0

                        desk_id = roi["desk"]
                        streak_key = (desk_id, person_idx)
                        current_detected.add(streak_key)
                        prediction_streak[streak_key] = prediction_streak.get(streak_key, 0) + 1 if pred == 1 else 0

                        # Draw labels
                        label = "Hand in Pocket" if pred else "No Hand in Pocket"
                        color = (0, 0, 255) if pred else (0, 255, 0)
                        cv2.putText(vis, f"{label} ({prob:.2f})", (int(max(0, person_x)), 50 + person_idx * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(vis, f"Desk: {desk_id}, Pos: {position}",
                                    (int(max(0, person_x)), 100 + person_idx * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 180, 0), 2)

                        if prediction_streak.get(streak_key, 0) >= ALERT_THRESHOLD:
                            print("*************ALERT**************")
                            alert_label = f"ALERT: {desk_id} - Person {person_idx}"
                            cv2.putText(vis, alert_label, (int(max(0, person_x)), 150 + person_idx * 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # Save row to CSV with the ACTUAL windowed values
                        row = {"frame": frame_number, "desk": desk_id}
                        for t in range(WINDOW_SIZE):
                            for i in range(10):
                                row[f"kp_{i}_x_t{t}"] = sliding_window[position][t][f"kp_{i}_x"]
                                row[f"kp_{i}_y_t{t}"] = sliding_window[position][t][f"kp_{i}_y"]
                        row["prediction"] = pred
                        row["probability"] = round(prob, 4)
                        csv_output.append(row)

            # reset streaks for keys not seen this frame
            for k in list(prediction_streak.keys()):
                if k not in current_detected:
                    prediction_streak[k] = 0

            cv2.imshow("MLP Inference", vis)
            if cv2.waitKey(WAITKEY) & 0xFF == ord('q'):
                break
        # VIDEO += 1
        # print(f"video num: {VIDEO}")
        cap.release()

        # Write CSV
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_csv = os.path.join(OUTPUT_DIR, os.path.splitext(video_file)[0] + ".csv")

        keypoint_cols = [f"kp_{i}_{axis}_t{t}"
                         for i in range(10)
                         for axis in ("x", "y")
                         for t in range(WINDOW_SIZE)]
        fieldnames = ["frame", "desk"] + keypoint_cols + ["prediction", "probability"]

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_output)

        # print(f"[OK] Wrote CSV: {out_csv}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
