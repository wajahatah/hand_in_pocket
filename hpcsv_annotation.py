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

def assign_roi_index(x):
    if x < 390:
        return -1
    elif 395 < x < 835:
        return 0
    # elif 840 < x < 1280:
    #     return 1
    else:
        return 2

if __name__ == "__main__":
    model = YOLO("C:/wajahat/hand_in_pocket/bestv7-2.pt")
    # input_dir = "C:/Users/LAMBDA THETA/Videos/new_class"
    input_dir = "F:/Wajahat/hand_in_pocket/Hands_in_pocket_tp/tp21-4"
    video_name = "c1_v9.mp4"
    output_dir = "C:/wajahat/hand_in_pocket/dataset/training"
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

    connections = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 7),
        (4, 5), (5, 6),
        (7, 8), (8, 9)
    ]

    csv_filename = os.path.join(output_dir, video_name)

    keypoint_headers = [f"kp_{i}_x" for i in range(10)] + [f"kp_{i}_y" for i in range(10)] + [f"kp_{i}_conf" for i in range(10)]
    distance_headers = [f"distance({i},{j})" for (i, j) in connections]
    headers = ["frame", "person_idx", "roi_idx", "position"] + keypoint_headers + distance_headers + ["hand_in_pocket"]

    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()

    frame_count = 0
    # roi_regions = [(0, 256), (256, 512), (512, 768), (768, 1024), (1024, 1280)]  # Adjust if needed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        results = model(frame)

        # detected_person = []
        person_info_list = []

        # Get user input for current frame
        # try:
        #     hand_in_pocket = int(input(f"[Frame {frame_count}] Enter hand_in_pocket (0/1): "))
        #     position = int(input(f"[Frame {frame_count}] Enter position (-2 to 2): "))
        # except ValueError:
        #     print("Invalid input. Skipping frame.")
        #     continue

        for result in results:
            keypoints = result.keypoints
            if keypoints is not None:
                keypoints_data = keypoints.data

                # for _, person_keypoints in enumerate(keypoints_data):
                for person_id, person_keypoints in enumerate(keypoints_data):
                    keypoint_list = []
                    # x_center = float(np.mean([kp[0].item() for kp in person_keypoints]))
                    # roi_idx = assign_roi_index(x_center)

                    row_data = {
                        "frame": frame_count}
                        # "roi_idx": roi_idx }
  
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
                    row_data["roi_idx"] = roi_idx

                    cv2.putText(frame, f"ROI: {roi_idx}", (int(roi_x), 50 + 30 * roi_idx), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    dist_dict = calculate_distances(keypoint_list, connections)
                    row_data.update(dist_dict)

                    person_info_list.append((roi_idx, row_data))

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        print(f"\n[Frame {frame_count}] has {len(person_info_list)} peoples.")

        for roi_idx, row_data in person_info_list:
            print(f"    ↪ Frame {frame_count} ")

            print(f"↪ Enter hand_in_pocket (0 or 1) for person (ROI {roi_idx}): ", end="")
            while True:
                hand_in_pocket = input()
                break
            
            print(f"↪ Enter position (-2 to 2) for person (ROI {roi_idx}): ", end="")
            while True:
                position = input()
                break

            row_data["hand_in_pocket"] = hand_in_pocket
            row_data["position"] = position

            csv_writer.writerow(row_data)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        frame_count += 1

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()


        #             # Assign ROI-based person_idx (based on nose x or fallback)
        #             base_x = keypoint_list[0][0] if keypoint_list[0][2] > 0.5 else sum([p[0] for p in keypoint_list]) / len(keypoint_list)
        #             person_idx = assign_roi_index(base_x)

        #             detected_person.append((person_idx, keypoint_list))
        
        
        # detected_person.sort(key=lambda x: x[0])  # Sort by person_idx

        # for person_idx, keypoint_list in detected_person:
        #     print(f"\n [frame {frame_count}] Detected person at ROI {person_idx}")

        #     try:
        #     except ValueError:
        #         print("Invalid input. Skipping this person.")
        #         continue

        #     row_data = {
        #         "frame": frame_count,
        #         "person_idx": person_idx,
        #         "hand_in_pocket": hand_in_pocket,
        #         "position": position
        #     }

        #     for i, (x, y, conf) in enumerate(keypoint_list):
        #         row_data[f"kp_{i}_x"] = x
        #         row_data[f"kp_{i}_y"] = y
        #         row_data[f"kp_{i}_conf"] = conf

        #     dist_dict = calculate_distances(keypoint_list, connections)
        #     row_data.update(dist_dict)

        #     csv_writer.writerow(row_data)

        # frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
        # cv2.imshow('Pose Detection', frame)
        # cv2.imwrite(frame_filename, frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # frame_count += 1

# okay one more thing I want to do, I have a json file in which all roi are present distinguish by the <"_id": "camera_1"> from each other, what I want is to get the 'xmin' and 'xmax' values from the file