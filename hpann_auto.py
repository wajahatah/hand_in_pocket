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
    input_dir = "F:/Wajahat/hand_in_pocket/Hands_in_pocket_tp"
    video_name = "v1.mp4"
    output_dir = "C:/wajahat/hand_in_pocket/dataset/training"
    json_path = "qiyas_multicam.camera_final.json"

    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    video_path = os.path.join(input_dir, video_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_width = 1280
    frame_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Show first frame to user
    ret, frame = cap.read()
    if not ret:
        print("Error reading first frame.")
        exit()

    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Select Camera View", frame)
    cv2.waitKey(1)

    print("Available cameras")
    with open(json_path, "r") as f:
        camera_config = json.load(f)
        for cam in camera_config:
            print(f"- {cam['_id']}")

    while True:
        camera_id_input = input("Enter camera ID for this video (e.g., camera_1): ")
        camera_id = f"camera_{camera_id_input}"

        camera_data = next((cam for cam in camera_config if cam["_id"] == camera_id), None)
        if camera_data:
            break
        print(f"Invalid camera ID: {camera_id}. Please try again.")

    cv2.destroyWindow("Select Camera View")

    # camera_id = input("\ud83d\udd0d Enter camera ID for this video (e.g., camera_1): ")
    # cv2.destroyWindow("Select Camera View")

    # with open(json_path, "r") as f:
    #     camera_config = json.load(f)

    # camera_data = next((cam for cam in camera_config if cam["_id"] == camera_id), None)
    # if camera_data is None:
    #     raise ValueError(f"\u274c No configuration found for camera ID: {camera_id}")

    # roi_data_list = camera_data["rois"]
    roi_data_list = list(camera_data["data"].values())
    roi_lookup = {roi["desk"]: roi for roi in roi_data_list}

    connections = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 7),
        (4, 5), (5, 6),
        (7, 8), (8, 9)
    ]

    csv_filename = os.path.join(output_dir, video_name + ".csv")

    keypoint_headers = [f"kp_{i}_x" for i in range(10)] + [f"kp_{i}_y" for i in range(10)] + [f"kp_{i}_conf" for i in range(10)]
    distance_headers = [f"distance({i},{j})" for (i, j) in connections]
    headers = ["frame", "person_idx", "position", "roi_idx", "xmin", "xmax"] + keypoint_headers + distance_headers + ["hand_in_pocket"]

    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()

    all_person_data = {}
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

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        print(f"\n[Frame {frame_count}] has {len(person_info_list)} people.")

        for row_data in person_info_list:
            roi_idx = row_data["roi_idx"]
            print(f"↪ Enter hand_in_pocket (0 or 1) for person (ROI {roi_idx}): ", end="")
            while True:
                hand_in_pocket = input()
                if hand_in_pocket in ["0", "1"]:
                    break
                print("❌ Invalid input. Please enter 0 or 1.")

            row_data["hand_in_pocket"] = hand_in_pocket
            csv_writer.writerow(row_data)

        frame_count += 1

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
