# This file contains all the post processing functions for our pipeline code.

"""
Funtions included:
        
"""
from .helper_functions_module import load_config
config = load_config('config_module.yaml')


def post_process_condition(person_idx, hand_id, condition, condition_buffer):
    """Post-process condition and update buffer."""
    condition_buffer[person_idx][hand_id].append(condition)

    # if condition == 1:
    #     print(f"Condition: {condition}")
    #     print(f"Condition Buffer: {condition_buffer}")

    if len(condition_buffer[person_idx][hand_id]) == config['CONDITION_BUFFER']:
        # Check if 2 or more values are 1 then condition is true
        if sum(condition_buffer[person_idx][hand_id]) > 1:

            return 1
        else:
            return 0
    else:
        # print("Not Enough frames for detection.")
        return 0
    
def update_condition_hold(person_idx, hand_id, condition_final, sitting_check, condition_hold):
    """Update condition hold based on conditions."""
    if condition_final == 1:  #and sitting_check == True:
        # print(f"Hand in Pocket for person {person_idx} for hand {hand_id}")
        condition_hold[person_idx][hand_id] = True
    return condition_hold

def update_condition_output(person_idx, hand_id, condition_hold, condition_output, condition_edge_output, desk_roi, hand_near_ear):
    """Update the final condition output."""

    if condition_hold[person_idx][hand_id]:

        # Check if the condition is for far right or left edge case
        if len(desk_roi) > 3:                           # Check if the legth of desk ROI is 4
            if person_idx == 1 and hand_id == 0:        # Check for Left hand of Left most desk for 4 desks
                condition_edge_output[person_idx-1] = condition_hold[person_idx][hand_id]
            elif person_idx == 4 and hand_id == 1:      # Check for Left hand of Left most desk for 4 desks
                condition_edge_output[person_idx-1] = condition_hold[person_idx][hand_id]

        # Check if the hand keypoint hides near ear keypoint
        if hand_near_ear == True:
            condition_edge_output[person_idx-1] = condition_hold[person_idx][hand_id]
        else:
            condition_output[person_idx-1] = condition_hold[person_idx][hand_id]

        # print(f"Condition_output: {condition_output}")
        # print(f"Condition_Edge_output: {condition_edge_output}")

    return condition_output, condition_edge_output

def check_for_condition_latch(hand_x, hand_y, angle_value, condition_hold, person_idx, hand_id):
    """Check if the condition latch should be brought out."""
    if (hand_x > 1 and hand_y > 1) or (angle_value < config['base_angle_threeshold_min'] or angle_value > config['base_angle_threeshold_max']):
        condition_hold[person_idx][hand_id] = False
        # print(f"Hand: {hand_x}, {hand_y}")
        # print("I am Here")
    return condition_hold

