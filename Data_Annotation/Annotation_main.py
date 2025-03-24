# This is out main file connecting all different files

import cv2
from pose.poseyolo import ypose
from hand_extractor.feature_extractor import Hand_in_Pocket_keypoints  # Import your first file class
from utils_module import helper_functions_module, csv_writer
import os
import re

# Example function to extract ROI for a specific camera and desk
def get_roi_for_camera(camera_id, roi_data):
    for camera in roi_data:
        if int(camera['_id'].split('_')[1]) == camera_id:
            return camera['data']
    return None


# Load config File
config = helper_functions_module.load_config('config_module_bulk.yaml')

# Initialize Hand_in_Pocket_keypoints
hand_in_pocket_model = Hand_in_Pocket_keypoints()

# Load ROI data (assuming it's loaded from a JSON file or directly assigned)
roi_data = helper_functions_module.load_roi_data("ROI_values/qiyas_multicam.camera.json")  # Your ROI JSON data here

# Retrieve ROI values for camera_id from the config
camera_id = hand_in_pocket_model.config_module['camera_id']
desk_roi = get_roi_for_camera(camera_id, roi_data)

# Initialize pose pipeline object
pose_pipeline = {'pose_pipeline': []}

if desk_roi is None:
    print(f"ROI data for camera_id {camera_id} not found.")
else:
    print(f"ROI data for camera_id {camera_id}: {desk_roi}")

# Initialize the pose model
pose_model = ypose(verbose=True)

# Open video capture (use the correct camera URL from ROI data)
camera_url = None
for camera in roi_data:
    if int(camera['_id'].split('_')[1]) == camera_id:
        camera_url = config['camera_url']
        break


# During your video processing loop
csv_initialized = False


if camera_url is None:
    print(f"Camera URL for camera_id {camera_id} not found.")
else:

    # Create a valid folder name from the camera URL by removing invalid characters
    def sanitize_folder_name(url):
        return re.sub(r'[^a-zA-Z0-9_-]', '_', url)

    # Create output directory structure based on the camera URL
    output_folder_name = sanitize_folder_name(camera_url.split('/')[-1])  # Clean folder name
    output_dir = os.path.join(config['output_base_dir'], output_folder_name)  # Adjust this if needed
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
    frame_counter = 0  # To count frames with data

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Get pose data
        frames, pose_pipeline = pose_model.pose([frame], pose_pipeline)  # Metadata can include frame info if needed

        # Pass frame, pose data, and ROI to the model and get csv values
        csv_values = hand_in_pocket_model.main(frame, pose_pipeline, {'desk_roi': desk_roi})
        print(f"CSV_values: {csv_values} ")
        
        for person_key, person_data in csv_values.items():
            if person_data:  # Check if data is not empty
                flat_data = csv_writer.flatten_dict(person_data)

                # Ensure 'camera_id' and 'class_output' are correctly added once
                flat_data['camera_id'] = camera_id
                flat_data['class_output'] = 0

                # Initialize CSV with header if needed
                if not csv_initialized:
                    csv_writer.initialize_csv(csv_file_path, flat_data)
                    csv_initialized = True

                # Write data row
                # csv_writer.write_to_csv(config['csv_file_path'], flat_data)
                print(f"csv output dir: {csv_output_dir}")
                csv_writer.write_to_csv(csv_file_path, flat_data)

                # Edit frame to write frame Number
                # Draw the frame number in the middle before saving
                frame_number_text = f"Frame: {frame_counter}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (0, 165, 255)  # Orange color
                thickness = 2
                text_size, _ = cv2.getTextSize(frame_number_text, font, font_scale, thickness)
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = (frame.shape[0] + text_size[1]) // 2

                cv2.putText(frame, frame_number_text, (text_x, text_y), font, font_scale, font_color, thickness)


                # Save the frame if data is written
                frame_counter += 1
                frame_filename = os.path.join(frame_output_dir, f"frame_{frame_counter:04d}.jpg")
                cv2.imwrite(frame_filename, frame)

        # Display the frame (optional)
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
