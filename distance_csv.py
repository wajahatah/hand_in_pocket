from ultralytics import YOLO
import os
import cv2
import numpy as np
import csv

# Avoid potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def draw_lines(frame, keypoints, connections):
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            x1, y1, conf1 = keypoints[start_idx]
            x2, y2, conf2 = keypoints[end_idx]
            if conf1 > 0.5 and conf2 > 0.5:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)  # Yellow lines

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


if __name__ == "__main__":
    # Load your trained YOLOv8 model
    input_dir = "G:/wajahat/hand_in_pocket/traiining"
    output_dir = "C:/wajahat/hand_in_pocket/dataset/training"
    model = YOLO("C:/wajahat/hand_in_pocket/bestv7-2.pt")

    # Open the video file
    video_path = "C:/Users/LAMBDA THETA/Videos/new_class/chunk_26-02-25_19-00.avi"
    # for video_file in os.listdir(input_dir):
    #     if not video_file.endswith(".mp4"):
    #         continue

    # video_name = os.path.splitext(video_file)[0]
    # video_path = os.path.join(input_dir, video_file)
    # print(f"Processing {video_name}...")
    cap = cv2.VideoCapture(video_path)

    # Get video properties for saving the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_with_keypoints.mp4', fourcc, fps, (frame_width, frame_height))
    # video_output_folder = os.path.join(output_dir, video_name)
    # frames_dir = os.path.join(video_output_folder, "frames")
    # os.makedirs(frames_dir, exist_ok=True)  # Create directory if it doesn't exist
    # out = cv2.VideoWriter(os.path.join(video_output_folder, f"{video_name}_output.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Define the skeleton connections (update these indices as needed)
    connections = [
        (0, 1), (0, 2), (0, 3),  # Keypoint 0 to 1, 2, and 3
        (1, 4), (1, 7),          # Keypoint 3 to 4 and 5
        (4, 5), (5, 6),          # Keypoint 5 to 6, and 6 to 9
        (7, 8), (8, 9)           # Keypoint 7 to 8, and 4 to 7
    ]

    # Prepare CSV file for distances output
    csv_filename = "distances_output_v1.csv"
    # csv_path = os.path.join(video_output_folder, f"{video_name}.csv")
    # header = ["frame", "person_idx"] + [f"distance({i},{j})" for (i, j) in connections]

    keypoint_headers = [f"kp_{i}_x" for i in range(10)] + [f"kp_{i}_y" for i in range(10)] + [f"kp_{i}_conf" for i in range(10)]
    distance_headers = [f"distance({i},{j})" for (i, j) in connections]
    headers = ["frame", "person_idx"] + keypoint_headers + distance_headers

    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
    csv_writer.writeheader()

    # Open a text file to save keypoints info (optional)
    # txt_file = open('keypoints_output.txt', 'w')

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0

    person_data = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to (1280, 720) for consistency
        frame = cv2.resize(frame, (1280, 720))

        # Run inference on the current frame
        results = model(frame)

        # Process each detected person
        for result in results:
            keypoints = result.keypoints
            if keypoints is not None:
                keypoints_data = keypoints.data
                for person_idx, person_keypoints in enumerate(keypoints_data):
                    keypoint_list = []
                    row_data = {"frame": frame_count, "person_idx": person_idx}

                    for kp_idx, keypoint in enumerate(person_keypoints):
                        x, y, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
                        keypoint_list.append((x, y, confidence))

                        # Store keypoint values in CSV row
                        row_data[f"kp_{kp_idx}_x"] = x
                        row_data[f"kp_{kp_idx}_y"] = y
                        row_data[f"kp_{kp_idx}_conf"] = confidence
                        # Draw keypoint
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                    # Optionally, draw skeleton lines on the frame
                    draw_lines(frame, keypoint_list, connections)

                    # Calculate distances for current person's keypoints
                    dist_dict = calculate_distances(keypoint_list, connections)
                    row_data.update(dist_dict)
                    # Create CSV row
                    # csv_row = {"frame": frame_count, "person_idx": person_idx}
                    # csv_row.update(dist_dict)
                    # csv_writer.writerow(csv_row)
                    # rows_to_write.append(csv_row)

                    # for row in rows_to_write:
                        # csv_writer.writerow(row)
                    
                    if person_idx not in person_data:
                        person_data[person_idx] = []
                    person_data[person_idx].append(row_data)
                    # person_data[person_idx].append({"frame": frame_count, "person_idx": person_idx, **dist_dict})


        # Write the processed frame to output video and save the frame as an image
        # out.write(frame) 
        # frame_filename = os.path.join("C:/wajahat/hand_in_pocket/dataset/test1/f1", f"frame_{frame_count:04d}.jpg")
        # os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

        frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    for person_idx in sorted(person_data.keys()):  # Sort to ensure order
        for row in person_data[person_idx]:
            csv_writer.writerow(row)
    # Release resources and close files
    cap.release()
    # out.release()
    # txt_file.close()
    csv_file.close()

    # print(f"CSV file '{csv_filename}' created with distances.")
    # print(f"Done: {video_name} → CSV + frames saved.")

    cv2.destroyAllWindows()