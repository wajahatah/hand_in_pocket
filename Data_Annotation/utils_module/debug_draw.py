# This function contains all the drawing functions for debugging purposes.

"""
This file contains the following functions:


"""

import cv2

def debug_hands(frame, person_idx, hand_id, hand_buffers, hand_sequence_value_buffer, elbow_sequence_value_buffer, hand_sequence_value, elbow_sequence_value):  

    if hand_id == 0 and person_idx == 1:
        debug_text_1 = f"Hand_Sequence Buffer {hand_id}: {hand_buffers[person_idx][hand_id]}"
        debug_text_2 = f"Hand_Sequence Value Buffer: {hand_sequence_value_buffer[person_idx][hand_id]}"
        debug_text_3 = f"Elbow_Sequence Value Buffer: {elbow_sequence_value_buffer[person_idx][hand_id]}"
        debug_text_4 = f"Hand_Sequence value: {hand_sequence_value}"
        debug_text_5 = f"Elbow_Sequence value: {elbow_sequence_value}"

        cv2.putText(frame, debug_text_1 , (10 ,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_2 , (10 ,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_3 , (10 ,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_4 , (10 ,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_5 , (10 ,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if hand_id == 1 and person_idx == 1:
        debug_text_1 = f"Hand_Sequence Buffer {hand_id}: {hand_buffers[person_idx][hand_id]}"
        debug_text_2 = f"Hand_Sequence Value Buffer: {hand_sequence_value_buffer[person_idx][hand_id]}"
        debug_text_3 = f"Elbow_Sequence Value Buffer: {elbow_sequence_value_buffer[person_idx][hand_id]}"
        debug_text_4 = f"Hand_Sequence value: {hand_sequence_value}"
        debug_text_5 = f"Elbow_Sequence value: {elbow_sequence_value}"

        cv2.putText(frame, debug_text_1 , (640 ,570), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_2 , (640 ,600), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_3 , (640 ,630), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_4 , (640 ,660), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_5 , (640 ,690), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def display_condition_info(frame, person_idx, hand_id, condition_output, condition_hold):

    """Display condition information on the frame."""
    if hand_id == 1 and person_idx == 0:
        put_text = f"condition_output : {condition_output}"
        cv2.putText(frame, put_text, (int(50), int(650)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        put_text = f"condition_hold : {condition_hold}"
        cv2.putText(frame, put_text, (int(50), int(600)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def debug_hands_new(frame, person_idx, hand_id, hand_sequence_value, elbow_sequence_value, ear_history ,hand_near_ear):  

    if hand_id == 0 and person_idx == 1:
        debug_text_1 = f"Hand_Sequence value: {hand_sequence_value}"
        debug_text_2 = f"Elbow_Sequence value: {elbow_sequence_value}"
        # debug_text_3 = f"Hand Ear: {ear_history} {hand_near_ear}"
        debug_text_3 = f"Hand Ear: {hand_near_ear}"

        cv2.putText(frame, debug_text_1 , (5 ,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_2 , (5 ,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_3 , (5 ,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if hand_id == 1 and person_idx == 1:
        debug_text_1 = f"Hand_Sequence value: {hand_sequence_value}"
        debug_text_2 = f"Elbow_Sequence value: {elbow_sequence_value}"
        # debug_text_3 = f"Hand Ear: {ear_history} {hand_near_ear}"
        debug_text_3 = f"Hand Ear: {hand_near_ear}"

        cv2.putText(frame, debug_text_1 , (5 ,630), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_2 , (5 ,660), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_3 , (5 ,690), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    if hand_id == 0 and person_idx == 2:
        debug_text_1 = f"Hand_Sequence value: {hand_sequence_value}"
        debug_text_2 = f"Elbow_Sequence value: {elbow_sequence_value}"
        # debug_text_3 = f"Hand Ear: {ear_history} {hand_near_ear}"
        debug_text_3 = f"Hand Ear: {hand_near_ear}"

        cv2.putText(frame, debug_text_1 , (320 ,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_2 , (320 ,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_3 , (320 ,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if hand_id == 1 and person_idx == 2:
        debug_text_1 = f"Hand_Sequence value: {hand_sequence_value}"
        debug_text_2 = f"Elbow_Sequence value: {elbow_sequence_value}"
        # debug_text_3 = f"Hand Ear: {ear_history} {hand_near_ear}"
        debug_text_3 = f"Hand Ear: {hand_near_ear}"

        cv2.putText(frame, debug_text_1 , (320 ,630), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_2 , (320 ,660), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_3 , (320 ,690), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    if hand_id == 0 and person_idx == 3:
        debug_text_1 = f"Hand_Sequence value: {hand_sequence_value}"
        debug_text_2 = f"Elbow_Sequence value: {elbow_sequence_value}"
        # debug_text_3 = f"Hand Ear: {ear_history} {hand_near_ear}"
        debug_text_3 = f"Hand Ear: {hand_near_ear}"

        cv2.putText(frame, debug_text_1 , (640 ,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_2 , (640 ,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_3 , (640 ,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if hand_id == 1 and person_idx == 3:
        debug_text_1 = f"Hand_Sequence value: {hand_sequence_value}"
        debug_text_2 = f"Elbow_Sequence value: {elbow_sequence_value}"
        # debug_text_3 = f"Hand Ear: {ear_history} {hand_near_ear}"
        debug_text_3 = f"Hand Ear: {hand_near_ear}"

        cv2.putText(frame, debug_text_1 , (640 ,630), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_2 , (640 ,660), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_3 , (640 ,690), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    if hand_id == 0 and person_idx == 4:
        debug_text_1 = f"Hand_Sequence value: {hand_sequence_value}"
        debug_text_2 = f"Elbow_Sequence value: {elbow_sequence_value}"
        # debug_text_3 = f"Hand Ear: {ear_history} {hand_near_ear}"
        debug_text_3 = f"Hand Ear: {hand_near_ear}"

        cv2.putText(frame, debug_text_1 , (960 ,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_2 , (960 ,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_3 , (960 ,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if hand_id == 1 and person_idx == 4:
        debug_text_1 = f"Hand_Sequence value: {hand_sequence_value}"
        debug_text_2 = f"Elbow_Sequence value: {elbow_sequence_value}"
        # debug_text_3 = f"Hand Ear: {ear_history} {hand_near_ear}"
        debug_text_3 = f"Hand Ear: {hand_near_ear}"

        cv2.putText(frame, debug_text_1 , (960 ,630), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_2 , (960 ,660), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, debug_text_3 , (960 ,690), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)