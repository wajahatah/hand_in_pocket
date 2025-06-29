""" load all frames, load json, load individual csvs and also the can laod the combined one, crop the frames by the roi add desk num in the name 
and sort them into two folders
check the comments to switch between individual csvs and combined csv
"""

import os
import cv2
import json
import pandas as pd
from tqdm import tqdm

# ======== CONFIG ========
frames_dir = ["/home/ubuntu/wajahat/hp/without_kp_frames/hp", "/home/ubuntu/wajahat/hp/without_kp_frames/no_hp"]
csv_dir = "/home/ubuntu/wajahat/hp/cnn_combine.csv"
roi_json_path = "/home/ubuntu/wajahat/hp/qiyas_multicam.camera_final.json"
output_dir = "/home/ubuntu/wajahat/hp/without_kp_crop"

os.makedirs(os.path.join(output_dir, "hand_in_pocket"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "no_hand_in_pocket"), exist_ok=True)

# ======== LOAD ROI JSON ========
with open(roi_json_path, 'r') as f:
    roi_list = json.load(f)

roi_data = {}
for cam_entry in roi_list:
    cam_id = cam_entry["_id"]  # e.g., camera_1
    roi_data[cam_id] = cam_entry["data"]  # all desks info per camera

# ======== To load all the individual csvs from the folder ========
# csv_index = {}
# for csv_file in os.listdir(csv_dir):
#     if csv_file.endswith("_keypoints.csv"):
#         video_id = os.path.splitext(csv_file)[0].replace("_keypoints", "")  # e.g., c1_v1
#         df = pd.read_csv(os.path.join(csv_dir, csv_file))
#         csv_index[video_id] = df.set_index(["frame", "desk_no"])
        # print("Available keys in csv_index:", list(csv_index.keys()))

# ======== To load single combined csv ========

df_all = pd.read_csv(csv_dir)

df_all.dropna(subset=['frame', 'desk_no', 'source_file'], inplace=True)
df_all['frame'] = df_all['frame'].astype(int)
df_all['desk_no'] = df_all['desk_no'].astype(int)
# df_all.set_index(['source_file', 'desk_no', 'frame'], inplace=True)


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
        source_file = f"{video_key}.csv" # for combined csv
        # print("video_key", video_key)
        cam_num = cam_id[1:]  # extract number from c1 → 1
        camera_key = f"camera_{cam_num}"

        # print("Available keys in csv_index:", list(csv_index.keys()))

        if camera_key not in roi_data:
            # print("roi not found")
            continue

        #uncomment this if need to use individual csvs
        # if video_key not in csv_index: # or camera_key not in roi_data:
        #     # print("not found")
        #     continue


        frame_img = cv2.imread(frame_path)
        if frame_img is None:
            continue

        roi_desks = roi_data[camera_key]
        # df = csv_index[video_key] # uncomment this if need to use individual csvs

        # print("Checking index uniqueness...")
        # if not df_all.index.is_unique:
        #     print("❌ Index is not unique!")
        #     dupes = df_all.index[df_all.index.duplicated()]
        #     print("Sample duplicates:\n", dupes[:10])
        # else:
        #     print("✅ Index is unique.")


        for region_id, desk_info in roi_desks.items():
            desk_num = desk_info["desk"]
            xmin, xmax = desk_info["xmin"], desk_info["xmax"]
            ymin, ymax = desk_info["ymin"], desk_info["ymax"]

            crop = frame_img[ymin:ymax, xmin:xmax]  # keep RGB
            crop_resized = cv2.resize(crop, (640, 640))

            # cv2.imshow("cropped roi", crop_resized)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            label = 0
            try:
                # row = df.loc[(frame_id, desk_num)] # uncomment this if need to use individual csvs
                # row = df_all.loc[(source_file, desk_num, frame_id)] # for combined csv
                row = df_all[
                    (df_all["source_file"] == source_file) &
                    (df_all["desk_no"] == desk_num) &
                    (df_all["frame"] == frame_id)
                ]

                if (row["hand_in_pocket"] == 1).any():
                    label = 1

                # if len(row) == 0:
                #     label = 0
                # elif (row["hand_in_pocket"] == 1).any():
                #     label = 1
                # else:
                #     label = 0

            except KeyError:
                pass  # No annotation means negative class

            out_name = f"{video_key}_d{desk_num}_f{frame_id}.jpg"
            out_path = os.path.join(
                output_dir,
                "hand_in_pocket" if label == 1 else "no_hand_in_pocket",
                out_name
            )
            cv2.imwrite(out_path, crop_resized)

    except Exception as e:
        print(f"Error processing {frame_path}: {e}")

cv2.destroyAllWindows()