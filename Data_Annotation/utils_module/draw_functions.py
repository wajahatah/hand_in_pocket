# This file contains all the helper functions for drawing

"""
This contains the following functions:

    Draw_lines: This function draws lines for connecting shoulder and elbow
    Draw_Triangle: Draw a traingle
    Draw Angles: This function draws angles

"""
import cv2
import numpy as np
import math

from .calculations import calulate_base_angle

def draw_lines(frame, keypoints, connections):
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            x1, y1, conf1 = keypoints[start_idx]
            x2, y2, conf2 = keypoints[end_idx]

            # Ensure confidence is above a threshold to draw
            if conf1 > 0.5 and conf2 > 0.5:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)  # Yellow lines

def draw_triangle(frame, vertex_1, vertex_2, vertex_3):
    x1, y1 = vertex_1
    x2, y2 = vertex_2
    x3, y3 = vertex_3

    cv2.line(frame, (int(x1), int(y1)), (int(x3), int(y3)), (0, 0, 255), 2)  # Red line
    cv2.line(frame, (int(x2), int(y2)), (int(x3), int(y3)), (0, 0, 255), 2)  # Red line
    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green hypotenuse

def draw_curves(frame, vertex_1, vertex_2, base_angle):
    x1, y1 = vertex_1
    x2, y2 = vertex_2

    hypotenuse_dx = abs(x2 - x1)
    hypotenuse_dy = abs(y2 - y1)

    # Radius of the arc (1/3 the length of hypotenuse)
    radius = int(math.sqrt(hypotenuse_dx**2 + hypotenuse_dy**2) / 3)

    # Center of the arc is the first vertex (Shoulder)
    center = (int(x1), int(y1))

    # Draw Curves Based on Position
    if x2>x1:       # For Right Shoulder
        if y2>y1:    
            cv2.ellipse(frame, center, (radius, radius), 0 , 0, base_angle, (0, 255, 255), 2)
        else:
            cv2.ellipse(frame, center, (radius, radius), 360-base_angle, 0, 0, (0, 255, 255), 2)

    # For Left Shoulder
    else:
        if y2>y1:
            cv2.ellipse(frame, center, (radius, radius), 0, 180 - base_angle, 180, (0, 255, 255), 2)
        else:
            cv2.ellipse(frame, center, (radius, radius), 0, 180, base_angle + 180, (0, 255, 255), 2)

    # Display the base angle text on the image
    angle_text = f"{base_angle:.2f}"
    angle_position = (int(x1 - 10), int(y1 - 10))  # Position to display the angle
        
    cv2.putText(frame, angle_text, angle_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


def draw_right_angle_triangle_with_angles(frame, vertex_1, vertex_2, vertex_3, keypoint_head, person_idx, hand_id, sequence_threshold, condition):

    # Ensure vertices are available
    if vertex_1 and vertex_2:

        # Step 2: Draw the triangle
        draw_triangle(frame, vertex_1, vertex_2, vertex_3)

        # Step 3: Calculate the Angle
        base_angle = calulate_base_angle(vertex_1 , vertex_2)

        # Optionally draw the right angle symbol at the vertex
        # right_angle_point = (int(x1), int(y1))
        # cv2.putText(frame, "90", right_angle_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Step 4: Draw the curve (angle arc)
        draw_curves(frame, vertex_1, vertex_2, base_angle)

        # Display Person IDs
        head_x, head_y = keypoint_head
        person_id_text = f'person_{person_idx}'
        person_id_position = (int(head_x), int(head_y + 10)) 

        cv2.putText(frame, person_id_text, person_id_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Display Hand IDs
        hand_id_text = f'hand_{hand_id}'
        elbow_x , elbow_y =  vertex_2
        hand_id_position = (int(elbow_x), int(elbow_y + 10))
        
        cv2.putText(frame, hand_id_text, hand_id_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Display Hand sequence Threshold
        hand_sequence_text = f'threshold: {sequence_threshold:.2f}'
        elbow_x , elbow_y =  vertex_2
        hand_sequence_position = (int(elbow_x), int(elbow_y -20))
        
        cv2.putText(frame, hand_sequence_text, hand_sequence_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if condition == True:
        # Display Alert Signal
            alert_id_text = f'hand in Pocket_{hand_id}'
            alert_id_position = (int(head_x), int(head_y - 40)) 

            cv2.putText(frame, alert_id_text, alert_id_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            

    return frame

def draw_debugmode (frame ,keypoint_head, hand_sequence_value_buffer, keypoint_list):

    # print(keypoint_list[4])

    shoulder_left_x, shoulder_left_y, conf = keypoint_list[4] 
    shoulder_right_x, shoulder_right_y, conf = keypoint_list[7]

    shoulder_mid_x = (shoulder_left_x + shoulder_right_x)/2 
    shoulder_mid_y = (shoulder_left_y + shoulder_right_y)/2

    shoulder_length = abs(shoulder_right_x - shoulder_left_x)
    
    
    # Display Sequence Buffer
    head_x, head_y, head_conf = keypoint_head
    buffer_value_text = f'buffer_{hand_sequence_value_buffer}'
    buffer_position = (int(head_x), int(head_y - 120)) 

    cv2.putText(frame, buffer_value_text, buffer_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Display Line Between shoulders and midpoint and length of the line
    line_value_text = f'length_{shoulder_length:.2f}'
    line_value_position = (int(shoulder_mid_x), int(shoulder_mid_y + 30)) 

    cv2.line(frame, (int(shoulder_left_x), int(shoulder_left_y)), (int(shoulder_right_x), int(shoulder_right_y)), (0, 0, 255), 2)
    cv2.circle(frame, (int(shoulder_mid_x), int(shoulder_mid_y)), 5, (0, 255,0), -1)
    cv2.putText(frame, line_value_text, line_value_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


    



