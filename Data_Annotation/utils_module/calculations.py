# This file contains all the helper functions for calculations

"""
This contains the following functions:

    calculate_base_angle: function is used to calculate base angle.
    calculate_vertices: This function is used to calculate vertices for our triangles.
    hand in pocket: This function is core detection for hand in pocket.
"""

import math
from .helper_functions_module import load_config

# Load Config File
config_module = load_config('config_module.yaml')

def calulate_base_angle(vertex_1, vertex_2):
    # Calculate the base angle (the angle between the hypotenuse and the base)
    # Hypotenuse vector (from first keypoint to second keypoint)
    x1, y1 = vertex_1
    x2, y2 = vertex_2

    hypotenuse_dx = abs(x2 - x1)
    hypotenuse_dy = abs(y2 - y1)

    # Calculate the angle between the hypotenuse and the base
    angle_rad = math.atan2(hypotenuse_dy, hypotenuse_dx) 
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    # print(p1,p2)
    return int(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

def calculate_vertices(keypoints, connections):

    vertices_list = []

    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            x1, y1 = keypoints[start_idx]
            x2, y2 = keypoints[end_idx]

            vertex_1 = (x1, y1)
            vertex_2 = (x2, y2)

            # Ensure confidence is above a threshold to calculate vertices
            if x1 > 1 and y1 > 1 and x2 >1 and x2 > 1:
                # Compute the third vertex of the right angle triangle
                third_vertex_x = x2
                third_vertex_y = y1
                vertex_3 = (int(third_vertex_x), int(third_vertex_y))
            else:
                vertex_1 = None
                vertex_2 = None
                vertex_3 = None
        
            vertices = (vertex_1, vertex_2, vertex_3)
            vertices_list.append(vertices)

    return vertices_list

# This function tries to identify and apporxiamte the chanes in angle based on desk position.
def angle_adjustment():
    pass

def hand_sequence_calculation(buffer):

    front_value = None
    back_value = None

    if len(buffer) > 1:  # Check if the list has more than 1 element
        # Find the first value greater than 0 from the front
        for i in range(len(buffer)):
            if buffer[i] > 0:
                front_value = buffer[i]
                break
        
        # Find the first value greater than 0 from the back
        for i in range(len(buffer)-1, -1, -1):
            if buffer[i] > 0:
                back_value = buffer[i]
                break
        
        # Calculate the difference
        if front_value and back_value:
            movement = back_value - front_value
            return round(movement)
        else:
            return 0
    else:
        return 0
    
def elbow_sequence_calculation(buffer, hand_id):

    front_value = None
    back_value = None

    if len(buffer) > 1:  # Check if the list has more than 1 element
        # Find the first value greater than 0 from the front
        for i in range(len(buffer)):
            if buffer[i] > 0:
                front_value = buffer[i]
                break
        
        # Find the first value greater than 0 from the back
        for i in range(len(buffer)-1, -1, -1):
            if buffer[i] > 0:
                back_value = buffer[i]
                break
        
        # Calculate the difference
        if front_value and back_value:
            movement = back_value - front_value

            if hand_id == 1:
                return round(movement)
            
            if hand_id == 0:
                return -round(movement)
        else:
            return 0

    else:
        return 0


def hand_in_pocket(keypoint_list, hand_point,vertex_1, vertex_2, hand_sequence_value, elbow_sequence_value, hand_id):

    """
    The function uses these factors:
        Hand Not Visible
        Base angle greater than some threeshold value.
        Hand_sequence
        Elbow Sequence
    """

    # Get vertex values
    x1, y1 = vertex_1
    x2, y2 = vertex_2


    # Initiliaze Values
    hand_visible = True
    hand_inside_pocket =  0
    base_angle = 0
    hand_sequence_flag = False
    base_angle_flag = False
    elbow_sequence_flag = False

    # hand Sequence Threshold
    hand_sequence_min = config_module ['hand_sequence_threshold']['hand_sequence_threshold_min']
    hand_sequence_max = config_module ['hand_sequence_threshold']['hand_sequence_threshold_max']

    # elbow Sequence Threshold
    elbow_sequence_min = config_module['elbow_sequence_threshold']['elbow_sequence_threshold_min']
    elbow_sequence_max = config_module['elbow_sequence_threshold']['elbow_sequence_threshold_max']

    hand_x, hand_y =  keypoint_list[hand_point]

    if hand_x < 1 and hand_y < 1:       # If the confidence is less than 0.5, the pipeline returns 0
        hand_visible = False

    if hand_sequence_value > hand_sequence_min and hand_sequence_value < hand_sequence_max:
        hand_sequence_flag = True

    if elbow_sequence_value > elbow_sequence_min and elbow_sequence_value < elbow_sequence_max:
        elbow_sequence_flag = True

    if y2>y1: # Only for Those Movements where elbow is below shoulder. This applies this condition.
        base_angle = calulate_base_angle(vertex_1, vertex_2)

    # if hand_visible == False and base_angle > config['base_angle_threeshold']:
    #     hand_inside_pocket = True

    # print(f"hand_visible: {hand_visible}, hand_sequence_flag: {hand_sequence_flag}")

    if base_angle > config_module['base_angle_threeshold_min'] and base_angle < config_module['base_angle_threeshold_max']:
        base_angle_flag = True

    if hand_sequence_flag == True and  hand_visible == False and base_angle_flag == True and elbow_sequence_flag == True:
        hand_inside_pocket = 1

    return hand_inside_pocket

def check_hands_near_to_ear(keypoint_list, ear_points, hand_points, ear_history, hand_id, person_idx):
    """
    Check if a given hand (left or right) is moving closer to the corresponding ear.
    
    Args:
        keypoints (list): List of detected keypoints.
        hand_id (int): 0 for left hand, 1 for right hand.
    
    Returns:
        bool: True if the hand is moving closer, otherwise False.
    """
    output_flag = False
    ear_x, ear_y = keypoint_list[ear_points]
    hand_x, hand_y = keypoint_list[hand_points]
    hand = (hand_x, hand_y)
    ear = (ear_x, ear_y)

    if hand != (0, 0):
        history = int(distance(hand, ear))
        ear_history[person_idx][hand_id] = history

    if hand == (0, 0):  # Hand disappears
        output_flag = ear_history[person_idx][hand_id] <= config_module['hand_near_ear']
    
    return output_flag, ear_history