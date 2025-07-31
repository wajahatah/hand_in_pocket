import os
import cv2
import csv
import json
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def parse_filename(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split('_')
    return parts[0], parts[1], int(parts[2][1:])  # c1, v1, f123

def assign_desk(x, y, roi_data):
    for roi_id, data in roi_data.items():
        if data["xmin"] <= x <= data["xmax"] and data["ymin"] <= y <= data["ymax"]:
            return roi_id, data
    return None, None

def load_roi_data(json_path):
    with open(json_path, 'r') as f:
        cam_data = json.load(f)
    camera_map = {}
    for cam in cam_data:
        cam_id = cam["_id"].split('_')[-1]
        camera_map[f"c{cam_id}"] = cam["data"]
    return camera_map

def split_continuous_sequences(frames, min_window=5):
    sequences = []
    current_seq = [frames[0]]

    for i in range(1, len(frames)):
        _, _, prev_f, _ = frames[i - 1]
        _, _, curr_f, _ = frames[i]
        if curr_f == prev_f + 1:
            current_seq.append(frames[i])
        else:
            if len(current_seq) >= min_window:
                sequences.append(current_seq)
            current_seq = [frames[i]]

    if len(current_seq) >= min_window:
        sequences.append(current_seq)
    return sequences

def process_temporal_sequences(image_dict, model, roi_data, output_csv):
    fieldnames = ["camera", "video", "frame", "desk"]
    for t in range(5):
        for kp in range(10):
            fieldnames += [f"kp_{kp}_x_t{t}", f"kp_{kp}_y_t{t}"]#, f"kp_{kp}_conf_t{t}"]
    fieldnames += ["position_a", "position_b", "position_c", "position_d", "hand_in_pocket"]

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for cam_vid, frames in image_dict.items():
            frames.sort(key=lambda x: x[2])  # sort by frame number
            continuous_sequences = split_continuous_sequences(frames)

            for sequence in continuous_sequences:
                for i in range(len(sequence) - 4):
                    window = sequence[i:i + 5]
                    all_keypoints = defaultdict(list)
                    desk_data = {}

                    valid_window = True
                    images = []
                    for _, _, _, path in window:
                        img = cv2.imread(path)
                        if img is None:
                            valid_window = False
                            break
                        images.append(img)
                    if not valid_window:
                        continue

                    for t, img in enumerate(images):
                        results = model(img, verbose=False)
                        keypoints_data = results[0].keypoints.data.cpu().numpy()

                        for person_kps in keypoints_data:
                            x, y, conf = person_kps[0][0], person_kps[0][1], person_kps[0][2]
                            if conf < 0.5:
                                x , y = 0, 0
                            desk_id, desk_info = assign_desk(x, y, roi_data.get(window[0][0], {}))
                            if desk_id is None:
                                continue
                            key = f"{desk_id}"
                            all_keypoints[key].append(person_kps)
                            if key not in desk_data:
                                desk_data[key] = desk_info

                    for desk_id, kp_list in all_keypoints.items():
                        if len(kp_list) != 5:
                            continue

                        d_info = desk_data[desk_id]
                        plist = d_info.get("position_list", [0, 0, 0, 0])
                        row = {
                            "camera": window[0][0],
                            "video": window[0][1],
                            "frame": window[0][2],
                            # "desk_no": desk_id,
                            "desk": d_info.get("desk", -1)
                        }
                        for t, kps in enumerate(kp_list):
                            for i in range(10):
                                row[f"kp_{i}_x_t{t}"] = float(kps[i][0])
                                row[f"kp_{i}_y_t{t}"] = float(kps[i][1])
                                # row[f"kp_{i}_conf_t{t}"] = float(kps[i][2])
                        row["position_a"], row["position_b"], row["position_c"], row["position_d"] = plist
                        row["hand_in_pocket"] = 0  # manual or default

                        writer.writerow(row)

if __name__ == "__main__":
    model_path = "bestv7-2.pt"
    image_folder = "F:/Wajahat/hand_in_pocket/frames/without_kp/no_hp"
    json_path = "qiyas_multicam.camera_final.json"
    output_csv = "F:/Wajahat/hand_in_pocket/frames/without_kp/n0_hp_annotations.csv"

    model = YOLO(model_path)
    roi_data = load_roi_data(json_path)

    image_dict = defaultdict(list)
    for fname in os.listdir(image_folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                c, v, f = parse_filename(fname)
                
                full_path = os.path.join(image_folder, fname)
                image_dict[(c, v)].append((c, v, f, full_path))
            except:
                print(f"Skipping invalid filename: {fname}")

    process_temporal_sequences(image_dict, model, roi_data, output_csv)
    print(f"✅ Temporal CSV saved to: {output_csv}")
