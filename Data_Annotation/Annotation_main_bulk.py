# This is our main file connecting all different files
import cv2
from pose.poseyolo import ypose
from hand_extractor.feature_extractor import Hand_in_Pocket_keypoints  # Import your first file class
from utils_module import helper_functions_module, csv_writer
import os


# Load config File
config = helper_functions_module.load_config('config_module.yaml')

# Base Dir
base_dir = config["base_dir"]  # Set this to your base directory path

# Initialize Hand_in_Pocket_keypoints
hand_in_pocket_model = Hand_in_Pocket_keypoints()

# Load ROI data (assuming it's loaded from a JSON file or directly assigned)
roi_data = helper_functions_module.load_roi_data("ROI_values/qiyas_multicam.camera.json")  # Your ROI JSON data here

# Initialize pose pipeline object
pose_pipeline = {'pose_pipeline': []}

# Iterate over the folders inside base dir for input testing
for camera_folder in os.listdir(base_dir):
    camera_folder_path = os.path.join(base_dir, camera_folder)
    
    if os.path.isdir(camera_folder_path) and camera_folder.startswith("cam_"):
        # Extract camera_id from folder name
        camera_id = int(camera_folder.split('_')[1])
        
        # Load ROI data for this camera_id
        desk_roi = helper_functions_module.get_roi_for_camera(camera_id, roi_data)
        
        if desk_roi is None:
            print(f"ROI data for camera_id {camera_id} not found.")
            continue
        else:
            print(f"ROI data for camera_id {camera_id}: {desk_roi}")

        # Initialize the pose model
        pose_model = ypose(verbose=False)
        
        # Process each video in the camera folder
        for video_file in os.listdir(camera_folder_path):
            video_path = os.path.join(camera_folder_path, video_file)
            
            if not video_file.lower().endswith(('.avi', '.mp4', '.mov')):  # Adjust extensions if needed
                continue
            
            camera_url = video_path  # Set camera_url to the video’s relative path

            # Create output directory structure
            output_folder_name = helper_functions_module.sanitize_folder_name(f"cam_{camera_id}_{os.path.splitext(video_file)[0]}")
            output_dir = os.path.join(config['output_base_dir'], output_folder_name)
            frame_output_dir = os.path.join(output_dir, 'frames')
            csv_output_dir = os.path.join(output_dir, 'csv')

            # Ensure directories exist
            os.makedirs(frame_output_dir, exist_ok=True)
            os.makedirs(csv_output_dir, exist_ok=True)

            # Path for the CSV file
            csv_file_path = os.path.join(csv_output_dir, f"{output_folder_name}_data.csv")

            # Initialize the video capture object
            cap = cv2.VideoCapture(camera_url)

            csv_initialized = False
            frame_counter = 0       # To count frames with data

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video or failed to grab frame from {camera_url}.")
                    break

                frames, pose_pipeline = pose_model.pose([frame], pose_pipeline)

                csv_values = hand_in_pocket_model.main(frame, pose_pipeline, {'desk_roi': desk_roi})
                # print(f"CSV_values: {csv_values}")

                for person_key, person_data in csv_values.items():
                    if person_data:
                        flat_data = csv_writer.flatten_dict(person_data)

                        # Ensure 'camera_id' and 'class_output' are correctly added once
                        flat_data['camera_id'] = camera_id
                        flat_data['class_output'] = 0

                        # Initialize CSV with header if needed
                        if not csv_initialized:
                            csv_writer.initialize_csv(csv_file_path, flat_data)
                            csv_initialized = True

                        
                        # Write data rows
                        csv_writer.write_to_csv(csv_file_path, flat_data)

                        frame_number_text = f"Frame: {frame_counter}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1
                        font_color = (0, 165, 255)
                        thickness = 2
                        text_size, _ = cv2.getTextSize(frame_number_text, font, font_scale, thickness)
                        text_x = (frame.shape[1] - text_size[0]) // 2
                        text_y = (frame.shape[0] + text_size[1]) // 2

                        cv2.putText(frame, frame_number_text, (text_x, text_y), font, font_scale, font_color, thickness)

                        frame_counter += 1
                        frame_filename = os.path.join(frame_output_dir, f"frame_{frame_counter:04d}.jpg")
                        cv2.imwrite(frame_filename, frame)

                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

cv2.destroyAllWindows()