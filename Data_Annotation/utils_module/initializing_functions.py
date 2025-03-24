# This file contains all the functions needed to initialize different buffers for our pipeline code.

"""
Funtions included:

"""
from collections import deque
from .helper_functions_module import load_config

config = load_config('config_module.yaml')


def initialize_condition_buffer(person_idx, condition_buffer):
    # Initialize Condition Buffer
    if person_idx not in condition_buffer:
        condition_buffer[person_idx] = {0: deque(maxlen=config['CONDITION_BUFFER'] ), 1: deque(maxlen=config['CONDITION_BUFFER'] )}  # 0: Left hand 1: right hand

def initialize_person_buffers(person_idx, hand_buffers, hand_sequence_value_buffer, elbow_buffers, elbow_sequence_value_buffer):
    if person_idx not in hand_buffers:
        # Initialize Hand Buffer
        hand_buffers[person_idx] = {0: deque(maxlen=config['HAND_BUFFER_SEQUENCE']), 1: deque(maxlen=config['HAND_BUFFER_SEQUENCE'])}  # 0: Left hand 1: right hand
        hand_sequence_value_buffer[person_idx] = {0: deque(maxlen=config['HAND_BUFFER_VALUE']), 1: deque(maxlen=config['HAND_BUFFER_VALUE'])}  # 0: Left hand 1: right hand
        
        # Initialize Elbow Buffers
        elbow_buffers[person_idx] = {0: deque(maxlen=config['ELBOW_BUFFER_SEQUENCE']), 1: deque(maxlen=config['ELBOW_BUFFER_SEQUENCE'])}  # 0: Left hand 1: right hand
        elbow_sequence_value_buffer[person_idx] = {0: deque(maxlen=config['ELBOW_BUFFER_VALUE']), 1: deque(maxlen=config['ELBOW_BUFFER_VALUE'])}  # 0: Left hand 1: right hand


def initialize_condition_hold(person_idx, condition_hold):
    if person_idx not in condition_hold:
        condition_hold[person_idx] = {0: False, 1: False}  # 0: Left hand 1: right hand.


def initialize_output_array(desk_roi):
    # Initialize an array with zeros based on the number of keys
    zero_array = [0] * len(desk_roi) # Mutliply by desk_roi length
    
    return zero_array


def hand_near_ear_values(person_idx, ear_history):
    if person_idx not in ear_history:
        ear_history[person_idx] = {0: False, 1: False}  # 0: Left hand 1: right hand.

def csv_dict(person_idx, csv_dict):
    if person_idx not in csv_dict:
        csv_dict[person_idx] = {}