""" The script is used to annotate hand in pocket videos and generating a csv file with the keypoints and distances values. 
It take camera number and video number as input and name the csv file to that number with camera number and video number as cN_vN."""

from ultralytics import YOLO
import os
import cv2
import numpy as np
import csv
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Global variable to hold loaded ROI data
roi_data_list = []

def draw_lines(frame, keypoints, connections):
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            x1, y1, conf1 = keypoints[start_idx]
            x2, y2, conf2 = keypoints[end_idx]
            if conf1 > 0.5 and conf2 > 0.5:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

def calculate_distances(keypoint_list, connections, conf_th=0.5):
    distances = {}
    for (i, j) in connections:
        if i < len(keypoint_list) and j < len(keypoint_list):
            x1, y1, conf1 = keypoint_list[i]
            x2, y2, conf2 = keypoint_list[j]
            if conf1 > conf_th and conf2 > conf_th:
                dist = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
            else:
                dist = float(0)
            distances[f"distance({i},{j})"] = dist
        else:
            distances[f"distance({i},{j})"] = 0.0
    return distances

def assign_roi_index(x):
    for roi in roi_data_list:
        if roi["xmin"] <= x < roi["xmax"]:
            return roi["desk"]
    return -1

if __name__ == "__main__":
    model = YOLO("C:/wajahat/hand_in_pocket/bestv7-2.pt")
    input_dir = "F:/Wajahat/looking_around_panic/may_7/Hands In Pocket/TP/"
    # video_name = "c2_v4"
    output_dir = "C:/wajahat/hand_in_pocket/dataset/training"
    json_path = "qiyas_multicam.camera_final.json"

    os.makedirs(output_dir, exist_ok=True)
    # frames_dir = os.path.join(output_dir, video_name,"frames")
    # os.makedirs(frames_dir, exist_ok=True)

    # video_path = os.path.join(input_dir, video_name + ".mp4")

    video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]
    if not video_files:
        print("No video files found in the input directory.")
        exit()

    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(input_dir, video_file)
        print(f"Processing {video_name}...")
    
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        frame_width = 1280
        frame_height = 720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        ret, frame = cap.read()
        if not ret:
            print("Error reading first frame.")
            exit()

        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Select Camera View", frame)
        cv2.waitKey(1)

        # print("Available cameras")
        with open(json_path, "r") as f:
            camera_config = json.load(f)
            # for cam in camera_config:
            #     print(f"- {cam['_id']}")

        while True:
            camera_id_input = input("Enter camera ID for this video (e.g., camera_1): ")
            video_num = input("Enter video name:")
            camera_id = f"camera_{camera_id_input}"
            camera_data = next((cam for cam in camera_config if cam["_id"] == camera_id), None)
            if camera_data:
                break
            print(f"Invalid camera ID: {camera_id}. Please try again.")

        cv2.destroyWindow("Select Camera View")

        roi_data_list = list(camera_data["data"].values())
        roi_lookup = {roi["desk"]: roi for roi in roi_data_list}

        connections = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 7),
            (4, 5), (5, 6),
            (7, 8), (8, 9)
        ]
        video_name = f"c{camera_id_input}_v{video_num}"
        csv_filename = os.path.join(output_dir, video_name + ".csv")

        keypoint_headers = [f"kp_{i}_x" for i in range(10)] + [f"kp_{i}_y" for i in range(10)] + [f"kp_{i}_conf" for i in range(10)]
        distance_headers = [f"distance({i},{j})" for (i, j) in connections]
        # headers = ["frame", "person_idx", "position", "roi_idx", "xmin", "xmax"] + keypoint_headers + distance_headers + ["hand_in_pocket"]
        headers = ["frame", "person_idx", "position", "desk_no"] + keypoint_headers + distance_headers + ["hand_in_pocket"]

        csv_file = open(csv_filename, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
        csv_writer.writeheader()

        all_frames_data = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            results = model(frame)
            person_info_list = []

            for result in results:
                keypoints = result.keypoints
                if keypoints is not None:
                    keypoints_data = keypoints.data

                    temp_person_info = []

                    for person_keypoints in keypoints_data:
                        keypoint_list = []
                        row_data = {"frame": frame_count}

                        for kp_idx, kp in enumerate(person_keypoints):
                            x, y, conf = kp[0].item(), kp[1].item(), kp[2].item()
                            keypoint_list.append((x, y, conf))
                            row_data[f"kp_{kp_idx}_x"] = x
                            row_data[f"kp_{kp_idx}_y"] = y
                            row_data[f"kp_{kp_idx}_conf"] = conf
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                        draw_lines(frame, keypoint_list, connections)

                        
                        roi_x = keypoint_list[0][0]
                        roi_idx = assign_roi_index(roi_x)
                        roi_data = roi_lookup.get(roi_idx)

                        if roi_data is None:
                            print(f"⚠️ No ROI config for roi_idx {roi_idx}, skipping.")
                            continue

                        cv2.putText(frame, f"ROI: {roi_idx}", (int(roi_x), 50 + 30 * roi_idx), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        row_data["desk_no"] = roi_idx
                        row_data["position"] = roi_data["position"]
                        # row_data["xmin"] = roi_data["xmin"]
                        # row_data["xmax"] = roi_data["xmax"]

                        dist_dict = calculate_distances(keypoint_list, connections)
                        row_data.update(dist_dict)

                        temp_person_info.append((roi_idx, row_data))


                    # Remap person_idx based on sorted roi
                    temp_person_info.sort(key=lambda x: x[0])
                    for new_idx, (_, row) in enumerate(temp_person_info):
                        row["person_idx"] = new_idx
                        person_info_list.append(row)

                    """
                    for person_id, person_keypoints in enumerate(keypoints_data):
                        keypoint_list = []
                        row_data = {"frame": frame_count, "person_idx": person_id}

                        for kp_idx, kp in enumerate(person_keypoints):
                            x, y, conf = kp[0].item(), kp[1].item(), kp[2].item()
                            keypoint_list.append((x, y, conf))
                            row_data[f"kp_{kp_idx}_x"] = x
                            row_data[f"kp_{kp_idx}_y"] = y
                            row_data[f"kp_{kp_idx}_conf"] = conf
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                        draw_lines(frame, keypoint_list, connections)

                        roi_x = keypoint_list[0][0]
                        roi_idx = assign_roi_index(roi_x)
                        roi_data = roi_lookup.get(roi_idx)

                        if roi_data is None:
                            print(f"⚠️ No ROI config for roi_idx {roi_idx}, skipping.")
                            continue

                        row_data["roi_idx"] = roi_idx
                        row_data["position"] = roi_data["position"]
                        row_data["xmin"] = roi_data["xmin"]
                        row_data["xmax"] = roi_data["xmax"]

                        dist_dict = calculate_distances(keypoint_list, connections)
                        row_data.update(dist_dict)

                        person_info_list.append(row_data)
    """
            all_frames_data.append((frame.copy(), person_info_list))
            # all_frames_data.roi((frame.copy(), person_info_list))
            frame_count += 1

        cap.release()

        # Rearranged annotation loop by person across all frames
        max_persons = max(len(info) for _, info in all_frames_data)
        print(max_persons)


        for person_idx in range(max_persons):
        # for roi_idx in range(max_persons):
            print(f"\n\u25ba Now annotating for Person #{roi_idx} of video {video_name}across all frames.")
            for frame_num, (frame, person_list) in enumerate(all_frames_data):
                if person_idx >= len(person_list):
                    continue
                row_data = person_list[person_idx]
                # row_data = person_list[roi_idx]
                frame_to_show = frame.copy()

                cv2.imshow("frame", frame_to_show)
                # cv2.imwrite( os.path.join(frames_dir, f"frame_{frame_num}.jpg", frame_to_show))
                cv2.waitKey(1)

                roi_idx = row_data["desk_no"]
                position = row_data["position"]
                prompt = f"Frame {frame_num} | ROI {roi_idx} (Position: {position}): Enter hand_in_pocket (0 or 1) [Default: 0]: "

                while True:
                    hand_in_pocket = input(prompt).strip()
                    if hand_in_pocket in ["", "0", "1"]:
                        hand_in_pocket = hand_in_pocket or "0"
                        break
                    print("❌ Invalid input. Please enter 0 or 1 or press Enter for default 0.")

                row_data["hand_in_pocket"] = hand_in_pocket
                csv_writer.writerow(row_data)

        print(f"Annotation completed and saved {video_name} CSV.")

    csv_file.close()
    cv2.destroyAllWindows()


