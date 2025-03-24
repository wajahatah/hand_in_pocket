# The base code is taken from Wajahat
# This is the code to predict Hand in Pocket Cases.

from utils_module import helper_functions_module, calculations, draw_functions, postprocessing, initializing_functions, debug_draw
from .create_csv_data import create_dict, calculate_all_distances, keypoints_list_to_dict

# Make a module
class Hand_in_Pocket_keypoints:
    def __init__(self) -> None:

        # Load Config File
        self.config_module = helper_functions_module.load_config('config_module.yaml')

        self.connections =  [(4,5), (7,8)]          # These are my connections for drawing lines between shoulder and elbow
        self.hand_points = [6,9]                    # These are my points for Hand Key Points
        self.head_point = 0                         # This is my head point
        self.elbow_points = [5,8]                   # These are my points for Hand Key Points
        self.ear_points = [1,2]                     # These are my points for Ear Key Points
        self.shoulder_points = [4,7]                # These are my points for shoulder Key Points

        # Initialize buffers dynamically based on person_idx
        self.hand_buffers = {}
        self.hand_sequence_value_buffer = {}
        self.elbow_buffers = {}
        self.elbow_sequence_value_buffer = {}

        # Buffer to store and hold hand in pocket condition
        self.condition_hold = {}
        self.condition_buffer = {}

        # Keypoints for Persons
        self.keypoints = {}

        # Dictionary to get all values for CSV
        self.csv_dict = {}

        # Ear Keypoints History for Check near ear visibility
        self.ear_history = {}   # History for ears

        print("******** Hand in Pocket Model Loaded Successfully ********")


    def initialize_keypoints(self,person_idx,):
        if person_idx not in self.keypoints:
            self.keypoints[person_idx] = None

    
    def main(self,frame,pose_object, roi_object):

        # print("pipeline_object: ",pipeline_object)

        # Get Length of Desk ROIs for initialization of array
        desk_roi = roi_object['desk_roi']
        # print(f'Desk_ROI: {desk_roi}')

        # Get Keypoints Dict from pipeline object
        keypoints_dict = pose_object['pose_data']

        # # Get Head detection Data from pipeline Object
        # head_detection = pipeline_object['head_detection']

        """
        Head detector is not necessary for ML Model. It can be used in the pipeline before running the ML as a acheck that if True, then proceed with ML detection
        
        """

        # Condition Output List
        condition_output = initializing_functions.initialize_output_array(desk_roi)         # Here we initialize our final output array

        # Condition Edge Output List
        condition_edge_output = initializing_functions.initialize_output_array(desk_roi)    # Here we initialize our final output array for Edge Cases

        for value in keypoints_dict:

            head_point = (value['Cx'], value['Cy'])
            ear_left_point = (value['Ax'], value['Ay'])
            ear_right_point = (value['Bx'], value['By'])
            shoulder_left_point = (value['Slx'], value['Sly'])
            shoulder_right_point = (value['Srx'], value['Sry'])
            elbow_left_point = (value['Elx'], value['Ely'])
            elbow_right_point = (value['Erx'], value['Ery'])

            hand_left_point = (value['Hlx'], value['Hly'])
            hand_right_point = (value['Hrx'], value['Hry'])

            person_idx = helper_functions_module.get_personid(head_point, desk_roi)

            keypoint_person = [head_point, ear_left_point , ear_right_point , (0,0), shoulder_left_point, elbow_left_point,
                        hand_left_point, shoulder_right_point, elbow_right_point, hand_right_point]
            
            self.keypoints[person_idx] = keypoint_person # This will be my list with keypoints for all the persons

        for person_idx, person_keypoints in self.keypoints.items():
            # print(f"Key_point for: {person_idx}")

            # Initialize buffers for new person dynamically
            initializing_functions.initialize_person_buffers(person_idx, self.hand_buffers, self.hand_sequence_value_buffer,
                                                                self.elbow_buffers, self.elbow_sequence_value_buffer)

            # Initialize Conditioin Hold for different Persons
            initializing_functions.initialize_condition_hold(person_idx, self.condition_hold)

            # Initialize Condition Buffer
            initializing_functions.initialize_condition_buffer(person_idx, self.condition_buffer)

            # Initialize Ear history for different Persons
            initializing_functions.hand_near_ear_values(person_idx, self.ear_history)

            # Initialize csv.dict dictionary
            initializing_functions.csv_dict(person_idx, self.csv_dict)
    
            # For Debugging
            # Print Keypoints
            # for kp_idx ,keypoint in enumerate(person_keypoints):
            #     x ,y  = keypoint
            #     # print(f"Keypoint {kp_idx}: (x={x:.2f}, y={y:.2f})")

            # Logic Goes Here. first it gets left shoulder then Right
            vertices_list = calculations.calculate_vertices(person_keypoints, self.connections)

            for hand_id, vertex in enumerate(vertices_list):
                vertex_1, vertex_2, vertex_3 = vertex

                if vertex_1 and vertex_2:

                    # Update buffer for hand keypoints
                    hand_x ,hand_y = person_keypoints[self.hand_points[hand_id]]
                    hand_y_rou = round(hand_y)
                    self.hand_buffers[person_idx][hand_id].append(hand_y_rou)

                    # Update buffer for elbow keupoints
                    elbow_x ,elbow_y = person_keypoints[self.elbow_points[hand_id]]
                    elbow_y_rou = round(elbow_y)
                    self.elbow_buffers[person_idx][hand_id].append(elbow_y_rou)

                    # Hand Sequence detection. Keep track of last {buffer} movement values. 
                    ## This is needed since at the instant hand goes into pocket, threshold value goes to zero.

                    # Update hand sequence value buffer
                    self.hand_sequence_value_buffer[person_idx][hand_id].append(calculations.hand_sequence_calculation(self.hand_buffers[person_idx][hand_id]))

                    # Update elbow sequence value buffer
                    self.elbow_sequence_value_buffer[person_idx][hand_id].append(calculations.elbow_sequence_calculation(self.elbow_buffers[person_idx][hand_id], hand_id))

                    # # Update hand near ear values only if hand points are visible
                    # if person_keypoints[self.hand_points[hand_id]] != (0,0):
                    #     self.ear_history[person_idx][hand_id].append(person_keypoints[self.ear_points[hand_id]])

                    # print(f"Here : {hand_sequence_value_buffer[hand_id]}")
                    # Get the maximum values for Hand and Elbow Distances from the Buffer
                    hand_sequence_value = max(self.hand_sequence_value_buffer[person_idx][hand_id])
                    elbow_sequence_value = max(self.elbow_sequence_value_buffer[person_idx][hand_id])

                    # Main Logic Starts from here

                    # First Check if the shoulder keypoints are inside ROI
                    inside_ROI_flag = helper_functions_module.check_inside_ROI(person_keypoints, self.shoulder_points[hand_id], desk_roi, person_idx)
                    # print(f"ROI_Flag {inside_ROI_flag}")

                    if inside_ROI_flag == True:
                        # Check condition for hand in pocket only for cases when inside ROI
                        condition = calculations.hand_in_pocket(person_keypoints, self.hand_points[hand_id], vertex_1, vertex_2,
                                                                    hand_sequence_value, elbow_sequence_value, hand_id)
                    else:
                        condition = 0

                    # Check if the hand hides near ear. Also updates history distance values
                    hand_near_ear, self.ear_history = calculations.check_hands_near_to_ear(person_keypoints, self.ear_points[hand_id], 
                                                                                           self.hand_points[hand_id], self.ear_history, 
                                                                                           hand_id, person_idx)

                    # Post processing
                    # condition_final = postprocessing.post_process_condition(person_idx, hand_id, condition, self.condition_buffer)
                    # if condition_final == 1:
                    #     print(f"Condition_Final: {condition_final}")

                    condition_final = condition

                    # Check if person is sitting
                    # sitting_check = helper_functions_module.check_sitting(person_keypoints[self.head_point], head_detection)

                    # Update condition hold
                    self.condition_hold = postprocessing.update_condition_hold(person_idx, hand_id, condition_final, None ,self.condition_hold)   # Remove sitting check
                    
                    # print(f"Condition_Hold: {self.condition_hold}")

                    # Get Angle value to use for holding condition
                    angle_value = int(calculations.calulate_base_angle(vertex_1, vertex_2))

                    # Check for condition latch
                    self.condition_hold = postprocessing.check_for_condition_latch(hand_x, hand_y, angle_value, self.condition_hold, person_idx, hand_id)

                    # Update condition output
                    condition_output, condition_edge_output = postprocessing.update_condition_output(person_idx, hand_id, self.condition_hold, 
                                                                                                     condition_output, condition_edge_output, 
                                                                                                     desk_roi, hand_near_ear)
                    
                    # Create an dictionary with difference bewteen each keypoint value
                    distances = calculate_all_distances(self.keypoints[person_idx])

                    # Create a dictionary for keypoints
                    keypoints_dict_csv = keypoints_list_to_dict(self.keypoints[person_idx])
                    
                    # Create a list of the required values that can be passed and used to create a dict
                    values = [person_idx, keypoints_dict_csv, distances, inside_ROI_flag, hand_near_ear, angle_value, hand_sequence_value, elbow_sequence_value]
                    
                    # Create a dictionary to output all values
                    if inside_ROI_flag == True and person_idx != None:
                        self.csv_dict[person_idx] = create_dict(self.csv_dict, values)
                    else:
                        self.csv_dict[person_idx] = {}

                    # Draw Functions for Debugging
                    if self.config_module['debug'] == True:
                        debug_draw.debug_hands_new(frame, person_idx, hand_id, hand_sequence_value, elbow_sequence_value, self.ear_history, hand_near_ear)
                        
                        # debug_draw.display_condition_info(frame, person_idx, hand_id, condition_output, condition_hold)

                        # Draw functions
                        draw_functions.draw_right_angle_triangle_with_angles(
                        frame, vertex_1, vertex_2, vertex_3, person_keypoints[self.head_point], person_idx, hand_id, hand_sequence_value,
                        self.condition_hold[person_idx][hand_id]
                        )

                        # print(f"Condition Hold: {self.condition_hold}")

        return  self.csv_dict

