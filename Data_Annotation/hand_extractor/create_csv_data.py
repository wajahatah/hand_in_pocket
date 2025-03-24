# This file is used to create an output dictionary object that can be used to write into csv

import math
from utils_module import calculations 

def create_dict(csv_dict , values):

    csv_dict = {
        'person_idx': values[0],
        'keypoints_dict_csv': values[1],
        'distances': values[2],
        'inside_ROI_flag': values[3],
        'hand_near_ear': values[4],
        'angle_value': values[5],
        'hand_sequence_value': values[6],
        'elbow_sequence_value': values[7],
    }

    return csv_dict


def calculate_all_distances(keypoints):
    """
    Calculate distances between all keypoint pairs and store them in a dictionary.
    
    Args:
        keypoints (dict): Dictionary containing keypoints with indices as keys and (x, y) tuples as values.
        
    Returns:
        dict: Dictionary with distance keys like 'lefthand_rightear' and their corresponding distances.
    """
    distance_dict = {}
    
    # Mapping keypoint indices to descriptive names
    point_names = {
        0: "head",
        1: "leftear",
        2: "rightear",
        4: "leftshoulder",
        5: "leftelbow",
        6: "lefthand",
        7: "rightshoulder",
        8: "rightelbow",
        9: "righthand"
    }
    
    points = [6, 9, 5, 8, 4, 7, 1, 2, 0]  # Order of points to calculate distances
    
    for i in range(len(points)):
        point1_idx = points[i]
        point1_name = point_names[point1_idx]
        point1_coords = keypoints[point1_idx]
        
        if point1_coords is None:
            continue
        
        for j in range(i + 1, len(points)):
            point2_idx = points[j]
            point2_name = point_names[point2_idx]
            point2_coords = keypoints[point2_idx]
            
            if point2_coords is None:
                continue

            dist = calculations.distance(point1_coords, point2_coords)
            key = f"{point1_name}_{point2_name}"

            if point1_coords == (0,0) or point2_coords == (0,0):
                distance_dict[key] = 0
            else:
                distance_dict[key] = dist
    
    return distance_dict

def keypoints_list_to_dict(keypoint_person):
    """
    Convert a list of keypoints into a dictionary with named keys.

    Parameters:
        keypoint_person (list of tuple): List of keypoints in the order:
            [head, ear_left, ear_right, (0,0), shoulder_left, elbow_left, hand_left,
             shoulder_right, elbow_right, hand_right]

    Returns:
        dict: Dictionary with keypoint names as keys and coordinates as values.
    """
    keypoints = {
        'head_x': keypoint_person[0][0],
        'head_y': keypoint_person[0][1],
        'ear_left_x': keypoint_person[1][0],
        'ear_left_y': keypoint_person[1][1],
        'ear_right_x': keypoint_person[2][0],
        'ear_right_y': keypoint_person[2][1],
        'shoulder_left_x': keypoint_person[4][0],
        'shoulder_left_y': keypoint_person[4][1],
        'elbow_left_x': keypoint_person[5][0],
        'elbow_left_y': keypoint_person[5][1],
        'hand_left_x': keypoint_person[6][0],
        'hand_left_y': keypoint_person[6][1],
        'shoulder_right_x': keypoint_person[7][0],
        'shoulder_right_y': keypoint_person[7][1],
        'elbow_right_x': keypoint_person[8][0],
        'elbow_right_y': keypoint_person[8][1],
        'hand_right_x': keypoint_person[9][0],
        'hand_right_y': keypoint_person[9][1]
    }
    
    return keypoints



