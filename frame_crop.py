""" load all frames, load json, load csv, crop the frames by the roi add desk num in the name and sort them into two folders"""

import os
import cv2
import json
import pandas as pd
from tqdm import tqdm

# ======== CONFIG ========
frames_dir = ["C:/wajahat/hand_in_pocket/dataset/without_kp/hp", "C:/wajahat/hand_in_pocket/dataset/without_kp/no_hp"]
csv_dir = "C:/wajahat/hand_in_pocket/dataset/split_keypoint"
roi_json_path = "qiyas_multicam.camera_final.json"
output_dir = "C:/wajahat/hand_in_pocket/dataset/without_kp_crop"

os.makedirs(os.path.join(output_dir, "hand_in_pocket"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "no_hand_in_pocket"), exist_ok=True)

# ======== LOAD ROI JSON ========
with open(roi_json_path, 'r') as f:
    roi_list = json.load(f)

roi_data = {}
for cam_entry in roi_list:
    cam_id = cam_entry["_id"]  # e.g., camera_1
    roi_data[cam_id] = cam_entry["data"]  # all desks info per camera

# ======== INDEX CSVs ========
csv_index = {}
for csv_file in os.listdir(csv_dir):
    if csv_file.endswith("_keypoints.csv"):
        video_id = os.path.splitext(csv_file)[0].replace("_keypoints", "")  # e.g., c1_v1
        df = pd.read_csv(os.path.join(csv_dir, csv_file))
        csv_index[video_id] = df.set_index(["frame", "desk_no"])
        # print("Available keys in csv_index:", list(csv_index.keys()))

# ======== PROCESS FRAMES ========
all_frames = []
for base_dir in frames_dir:
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".jpg"):
                all_frames.append(os.path.join(root, f))

for frame_path in tqdm(sorted(all_frames)):
    try:
        frame_name = os.path.basename(frame_path)
        parts = frame_name.split("_")
        cam_id = parts[0]  # c1
        vid_id = parts[1]  # v1
        frame_str = parts[2].split(".")[0]  # f0000
        frame_id = int(frame_str[1:])
        # print("part:", parts, "cam:", cam_id, "vid:", vid_id)

        video_key = f"{cam_id}_{vid_id}"
        # print("video_key", video_key)
        cam_num = cam_id[1:]  # extract number from c1 → 1
        camera_key = f"camera_{cam_num}"

        # print("Available keys in csv_index:", list(csv_index.keys()))

        if camera_key not in roi_data:
            # print("roi not found")
            continue
        if video_key not in csv_index: # or camera_key not in roi_data:
            # print("not found")
            continue


        frame_img = cv2.imread(frame_path)
        if frame_img is None:
            continue

        roi_desks = roi_data[camera_key]
        df = csv_index[video_key]

        for region_id, desk_info in roi_desks.items():
            desk_num = desk_info["desk"]
            xmin, xmax = desk_info["xmin"], desk_info["xmax"]
            ymin, ymax = desk_info["ymin"], desk_info["ymax"]

            crop = frame_img[ymin:ymax, xmin:xmax]  # keep RGB
            crop_resized = cv2.resize(crop, (640, 640))

            cv2.imshow("cropped roi", crop_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            label = 0
            try:
                row = df.loc[(frame_id, desk_num)]
                if row["hand_in_pocket"] == 1:
                    label = 1
            except KeyError:
                pass  # No annotation means negative class

            out_name = f"{video_key}_d{desk_num}_{frame_id}.jpg"
            out_path = os.path.join(
                output_dir,
                "hand_in_pocket" if label == 1 else "no_hand_in_pocket",
                out_name
            )
            cv2.imwrite(out_path, crop_resized)

    except Exception as e:
        print(f"Error processing {frame_path}: {e}")

cv2.destroyAllWindows()