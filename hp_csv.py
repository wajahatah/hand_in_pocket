from ultralytics import YOLO
import os
import cv2
import numpy as np
import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

# === Main Code ===
input_dir = "G:/wajahat/hand_in_pocket/traiining"
output_dir = "C:/wajahat/hand_in_pocket/dataset/training"
model = YOLO("C:/wajahat/hand_in_pocket/bestv7-2.pt")

connections = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (1, 7),
    (4, 5), (5, 6),
    (7, 8), (8, 9)
]

keypoint_headers = [f"kp_{i}_x" for i in range(10)] + [f"kp_{i}_y" for i in range(10)] + [f"kp_{i}_conf" for i in range(10)]
distance_headers = [f"distance({i},{j})" for (i, j) in connections]
headers = ["frame", "person_idx"] + keypoint_headers + distance_headers

# Loop through each video
for video_file in os.listdir(input_dir):
    if not video_file.endswith(".mp4"):
        continue

    video_name = os.path.splitext(video_file)[0]
    video_path = os.path.join(input_dir, video_file)
    print(f"Processing {video_name}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_file}")
        continue

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Prepare output paths
    video_output_folder = os.path.join(output_dir, video_name)
    frames_dir = os.path.join(video_output_folder, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    out_video_path = os.path.join(video_output_folder, f"{video_name}_output.mp4")
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    csv_path = os.path.join(output_dir, f"{video_name}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()

    frame_count = 0
    person_data = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        results = model(frame)

        for result in results:
            keypoints = result.keypoints
            if keypoints is not None:
                keypoints_data = keypoints.data
                for person_idx, person_keypoints in enumerate(keypoints_data):
                    keypoint_list = []
                    row_data = {"frame": frame_count, "person_idx": person_idx}

                    for kp_idx, keypoint in enumerate(person_keypoints):
                        x, y, conf = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
                        keypoint_list.append((x, y, conf))
                        row_data[f"kp_{kp_idx}_x"] = x
                        row_data[f"kp_{kp_idx}_y"] = y
                        row_data[f"kp_{kp_idx}_conf"] = conf
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                    draw_lines(frame, keypoint_list, connections)
                    row_data.update(calculate_distances(keypoint_list, connections))

                    if person_idx not in person_data:
                        person_data[person_idx] = []
                    person_data[person_idx].append(row_data)

        # Save frame
        out.write(frame)
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Write CSV
    for person_idx in sorted(person_data.keys()):
        for row in person_data[person_idx]:
            csv_writer.writerow(row)

    cap.release()
    out.release()
    csv_file.close()

    print(f"Done: {video_name} → CSV + frames saved.")

cv2.destroyAllWindows()
