"""import os
import cv2
import csv
import ast
import pandas as pd
from ultralytics import YOLO

# Frame dimensions (replace with actual)
frame_width = 1280
frame_height = 720
subtraction_values = {"person2": 370, "person3": 740, "person4": 1080}
body_parts = ["head", "neck", "left_ear", "right_ear", "left_shoulder", "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"]

# Person bounding ranges
ranges = [
    {'xmin': 60, 'ymin': 90, 'xmax': 367, 'ymax': 400},
    {'xmin': 370, 'ymin': 90, 'xmax': 730, 'ymax': 400},
    {'xmin': 740, 'ymin': 90, 'xmax': 1075, 'ymax': 400},
    {'xmin': 1080, 'ymin': 90, 'xmax': 1280, 'ymax': 400}
]

def parse_and_normalize(x, y, confidence, person):
    if confidence == 0:
        return (0, 0)
    
    # Apply subtraction and normalization
    subtraction = subtraction_values.get(person, 0)
    x -= subtraction
    x_norm = x / frame_width
    y_norm = y / frame_height
    return (x_norm, y_norm)

if __name__ == "__main__":
    # Load YOLO model
    model = YOLO("runs/pose/trail4/weights/best_y11.pt")
    video_path = "C:/Users/LAMBDA THETA/Downloads/handraise/cb4-65.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    output_folder = "frames_output/video1"
    os.makedirs(output_folder, exist_ok=True)
    output_csv_path = os.path.join(output_folder, "cb4-65.csv")

    # Prepare CSV file with headers
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "person"] + body_parts)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1280, 720))
        results = model(frame)

        person_columns = [[] for _ in range(4)]
        for result in results:
            keypoints = result.keypoints
            if keypoints is not None:
                for person_keypoints in keypoints.data:
                    for keypoint in person_keypoints:
                        x, y, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
                        for idx, r in enumerate(ranges):
                            if r['xmin'] <= x <= r['xmax'] and r['ymin'] <= y <= r['ymax']:
                                normalized = parse_and_normalize(x, y, confidence, f"person{idx+1}")
                                person_columns[idx].append(normalized)
                                break

        with open(output_csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for i, person_data in enumerate(person_columns):
                person = f"person{i+1}"
                if person_data:
                    formatted_data = [f"({x:.2f},{y:.2f})" for x, y in person_data[:10]]
                else:
                    formatted_data = ["(0.00,0.00)"] * 10

                writer.writerow([frame_count, person] + formatted_data)

        frame_count += 1

    cap.release()
    print(f"Combined CSV saved at {output_csv_path}.")"""

from ultralytics import YOLO
import os
import cv2
import csv
import numpy as np

# Avoid potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define the x-coordinate ranges for each person
# person_ranges = {
#     "Person1": {"x": (60, 367), "y":(90,580)},
#     "Person2": {"x": (370, 730), "y":(90,580)},
#     "Person3": {"x": (740, 1075), "y":(90,580)},
#     "Person4": {"x": (1080, 1280), "y":(90,580)}
# }

person_ranges = {
    "Person1": lambda x: 90 < x < 367,
    "Person2": lambda x: 370 <= x <= 680,
    "Person3": lambda x: 680 <= x <= 1030,
    "Person4": lambda x: x > 1030
}

# Keypoint labels
keypoint_labels = ["head", "neck", "left_ear", "right_ear", "left_shoulder",
                   "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"]



""" for video inference 

# output_dir = "C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/frames_output"
# frames_output_dir = os.path.join(output_dir, "v29")
# os.makedirs(frames_output_dir, exist_ok=True)

if __name__ == "__main__":
    # Load your trained YOLOv8 model
    model = YOLO("runs/pose/trail4/weights/best_y11.pt")

    # Open the video file
    video_path = "C:/Users/LAMBDA THETA/Downloads/handraise/11.mp4" #"Cam_19_14.mp4"
    cap = cv2.VideoCapture(video_path)

    # Get video properties for saving the output video
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    # # out = cv2.VideoWriter('output_with_keypoints.mp4', fourcc, fps, (frame_width, frame_height))
    # output_video_path = os.path.join(output_dir, 'v8.mp4')
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Open a CSV file to save keypoints
    # output_csv_path = os.path.join(frames_output_dir, 'v29s.csv')
    # with open(output_csv_path, 'w', newline='') as csvfile:
        # csv_writer = csv.writer(csvfile)

        # Write header row for the CSV
        # header = ["Frame"] + [f"{person}_{kp}" for person in person_ranges.keys() for kp in keypoint_labels]
        # csv_writer.writerow(header)

        # Check if the video capture opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0  # To track frame number

    # Read and process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available

        # Run inference on the current frame
        frame = cv2.resize(frame, (1280, 720))
        results = model(frame)
        # black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)


        # Create a dictionary to store keypoints for each person in the current frame
        # frame_keypoints = {person: ["(0,0,0.0)"] * len(keypoint_labels) for person in person_ranges.keys()}

        # Iterate over each detected object and process their keypoints
        for result in results:
            keypoints = result.keypoints  # Access the keypoints object

            if keypoints is not None:
                # Get the data attribute, which contains x, y, and confidence values
                keypoints_data = keypoints.data
                for person_keypoints in keypoints_data:
                    for kp_idx, keypoint in enumerate(person_keypoints[:len(keypoint_labels)]):
                        x, y, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()

                        # Determine which person the keypoint belongs to based on x-coordinate ranges
                        # for person, condition in person_ranges.items():
                            # if condition(x):
                                # frame_keypoints[person][kp_idx] = f"({x:.2f},{y:.2f},{confidence:.2f})"
                                # break
                    # x_mean = np.mean(person_keypoints[:, 0].cpu().numpy())
                    
                    # # Determine which person the keypoints belong to
                    # person_label = None
                    # for person, (x_min, x_max) in person_ranges.items():
                    #     if x_min <= x_mean <= x_max:
                    #         person_label = person
                    #         break

                    # if person_label:
                    #     for kp_idx, keypoint in enumerate(person_keypoints[:len(keypoint_labels)]):
                    #         x, y, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
                    #         frame_keypoints[person_label][kp_idx] = f"({x:.2f},{y:.2f},{confidence:.2f})"

                            # Draw a circle at each keypoint
                        cv2.circle(frame, (int(x), int(y)), 7, (0, 214, 31), -1)

                            # Put the keypoint values on the frame
                        # cv2.putText(
                        #     frame,
                        #     f"({int(x)}, {int(y)})",
                        #     (int(x) + 5, int(y) - 5),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.4,
                        #     (255, 0, 0),
                        #     1
                        # )

            # Write the processed frame keypoints to the CSV
            # row = [frame_count] + [kp for person in person_ranges.keys() for kp in frame_keypoints[person]]
            # csv_writer.writerow(row)

            # Write the processed frame to the output video file
            # out.write(frame)
            frame_filename = os.path.join(frames_output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            # cv2.imwrite(frame_filename, black_frame)


            # Display the frame with keypoints and values
            cv2.imshow('Pose Detection', frame)

            # Delay of 50 milliseconds to slow down the video playback
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

            frame_count += 1  # Increment frame counter

    # Release the video capture and close display window
    cap.release()
    # out.release()
    cv2.destroyAllWindows()


print(f"Transformed data saved to {output_dir}")
"""

""" for image inference"""

input_dir = "C:/Users/LAMBDA THETA/Downloads/train/train/images"
output_dir = "C:/Users/LAMBDA THETA/Downloads/train/train/ann_images"


if __name__ == "__main__":
    # Ensure output directory exists
    model = YOLO("runs/pose/trail4/weights/best_y11.pt")
    
    os.makedirs(output_dir, exist_ok=True)

    # Process each folder in the input directory
        # input_folder_path = os.path.join(input_dir, folder_name)
        # frame_path = os.path.join(input_dir, frame_name)
        # if not os.path.isdir(input_folder_path):
            # continue  # Skip non-folder files

    output_folder_path = os.path.join(output_dir)
    os.makedirs(output_folder_path, exist_ok=True)

        # Prepare CSV file to store keypoints
        # output_csv_path = os.path.join(output_folder_path, f"{folder_name}_keypoints.csv")
        # with open(output_csv_path, 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     # Write header row
        #     header = ["Frame"] + [f"{person}_{kp}" for person in person_ranges.keys() for kp in keypoint_labels]
        #     csv_writer.writerow(header)

            # Process each frame in the folder
        # for frame_name in sorted(os.listdir(input_folder_path)):
    for frame_name in sorted(os.listdir(input_dir)):
        frame_path = os.path.join(input_dir, frame_name)
        if not frame_name.lower().endswith(('.png', '.jpg', '.jpeg','.PNG')):
            continue  # Skip non-image files

        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (1280, 720))  # Resize to fixed dimensions
        results = model(frame)

            # Dictionary to store keypoints for each person
            # frame_keypoints = {person: ["(0,0,0.0)"] * len(keypoint_labels) for person in person_ranges.keys()}

            # Process detected keypoints
        for result in results:
            keypoints = result.keypoints
            if keypoints is not None:
                keypoints_data = keypoints.data
                for person_keypoints in keypoints_data:
                    for keypoint in person_keypoints:
                        x, y, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
                        cv2.circle(frame, (int(x), int(y)), 7, (0, 255, 0), -1)
                    # x_mean = np.mean(person_keypoints[:, 0].cpu().numpy())

                    # # Determine person label based on x-coordinate
                    # person_label = None
                    # for person, condition in person_ranges.items():
                    #     if condition(x_mean):
                    #         person_label = person
                    #         break

                    # if person_label:
                    #     for kp_idx, keypoint in enumerate(person_keypoints[:len(keypoint_labels)]):
                            # frame_keypoints[person_label][kp_idx] = f"({x:.2f},{y:.2f},{confidence:.2f})"
                            # Draw keypoints on the frame

        # Save keypoints to CSV
        # row = [frame_name] + [kp for person in person_ranges.keys() for kp in frame_keypoints[person]]
        # csv_writer.writerow(row)

        # Save the annotated frame
        output_frame_path = os.path.join(output_folder_path, frame_name)
        cv2.imwrite(output_frame_path, frame)

        print(f"Processed and saved: {output_frame_path}")

    print(f"Processing completed. Outputs saved to {output_dir}.")