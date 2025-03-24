# This file contains all the helper functions for our pipeline code.

"""
Funtions included:
        load_config: Load YAML files (our config files)
"""
import yaml
import json
import re

# Load configuration from YAML file
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_personid(head_point, desk_roi_data):
    # Get the 'desk_roi' data
    head_x, head_y = head_point
    
    # Iterate through each desk's ROI to check if the point (x, y) is inside
    for desk_key, roi in desk_roi_data.items():
        if roi['xmin'] <= head_x <= roi['xmax'] and roi['ymin'] <= head_y <= roi['ymax']:
            return int(desk_key)  # Return the desk number (key)
    
    return None  # Return None if point is not inside any desk

def check_sitting(head_point, head_detection_data):
    # Get the 'desk_roi' data
    head_values = head_detection_data
    head_x, head_y = head_point

    # print(f"Head Values: {head_values}")
    # print(f"Head Point: {head_point}")

    for value in head_values:
        x_min, y_min, x_max, y_max = value

        if  x_min<= head_x <= x_max and y_min <= head_y <= y_max:
            return True
    
    return False  # Return None if point is not inside any desk


def check_inside_ROI(person_keypoints, shoulder_point, desk_roi_data, person_idx):

    """
    This function checks if the shoulder keypoints are inside ROI
    
    Parameters:
    keypoints (list of tuples): List of keypoints (x, y) for a person.
    desk_roi_data (dict): A dictionary containing the desk ROI information.
    
    Returns:
    bool: True if the shoulder keypoints are inside the ROI, False otherwise.
    """
    # Iterate through each keypoint in the list
    shoulder_x , shoulder_y = person_keypoints[shoulder_point]

    # Check if the point is inside the relevant desk ROI
    is_inside = False

    for desk_key, roi in desk_roi_data.items():
        if int(desk_key) == person_idx:
            if roi['xmin'] <= shoulder_x <= roi['xmax'] and roi['ymin'] <= shoulder_y <= roi['ymax']:
                is_inside = True
                break  # No need to check further if we found a matching ROI
    if not is_inside:
        return False  # If any keypoint is outside ROI, return False
    
    return True  # All keypoints are inside the ROI

def load_roi_data(file_path):
    try:
        with open(file_path, 'r') as file:
            roi_data = json.load(file)
        print("ROI data loaded successfully.")
        return roi_data
    except Exception as e:
        print(f"Error loading ROI data: {e}")
        return None

# Example function to extract ROI for a specific camera and desk
def get_roi_for_camera(camera_id, roi_data):
    for camera in roi_data:
        if int(camera['_id'].split('_')[1]) == camera_id:
            return camera['data']
    return None

def sanitize_folder_name(url):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', url)
