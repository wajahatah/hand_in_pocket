from ultralytics import YOLO
import os
import cv2
import numpy as np
import joblib
import pandas as pd
from collections import deque
import json

# Avoid potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

roi_data_list = []

def assign_roi_index(x):
    for roi in roi_data_list:
        if roi["xmin"] <= x < roi["xmax"]:
            return roi["position"]
        
        # else:
        #     cv2.putText(frame, "Position not detected", (int(600), int(300)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # return -5

# Load models
if __name__ == "__main__":
    kp_model = YOLO("C:/wajahat/hand_in_pocket/bestv7-2.pt")
    rf_model = joblib.load("rf_models/rf_grid_temp_norm_l1_v2.joblib")  # your temporal model

    # video_path = "C:/wajahat/hand_in_pocket/test_bench/tp_t2.mp4"
    # input_dir = "C:/wajahat/hand_in_pocket/test_bench"
    input_dir = "C:/Users/LAMBDA THETA/Videos"
    json_path = "qiyas_multicam.camera_final.json"
    # json_path = "qiyas_multicam_2.camera.json"

    video_files = [ f for f in os.listdir(input_dir) if f.endswith('.mp4') ]
    if not video_files:
        print("No video files found in the directory.")
        exit()

    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        print(f"Processing video: {video_path}")
    #"F:/Wajahat/hand_in_pocket/hand_in_pocket/cam_1/chunk_26-02-25_10-15-desk1-2-3.avi"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            continue

        # Connections and feature names
        connections = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 7), (4, 5), (5, 6), (7, 8), (8, 9)]

        single_frame_features = [ "position" ]  + [
            f"kp_{i}_x" for i in range(10)
        ] + [
            f"kp_{i}_y" for i in range(10)
        ] 
        # + [
        #     "distance(0,1)", "distance(0,2)", "distance(0,3)", "distance(1,4)", "distance(1,7)",
        #     "distance(4,5)", "distance(5,6)", "distance(7,8)", "distance(8,9)", "position"
        # ]

        # Define temporal window size
        WINDOW_SIZE = 5
        # sliding_window = deque(maxlen=WINDOW_SIZE)
        sliding_window = {}

                
        # while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))

        # if roi_data_list == []:

        cv2.imshow("Select Camera", frame)
        cv2.waitKey(1)
            
        with open(json_path, 'r') as f:
            camera_config = json.load(f)

        skip_video = False
        while True:
            camera_id_input = input("Enter camera ID: ")
             
            if camera_id_input.lower() == 's':
                print(f"Skipping video {video_path}")
                cap.release()
                cv2.destroyWindow("Select Camera")
                skip_video = True
                break

            else: 
                camera_id = f"camera_{camera_id_input}"
                camera_data = next((cam for cam in camera_config if cam["_id"] == camera_id), None)
                if camera_data:
                    break
                print(f"Invalid camera ID {camera_id}. Please try again.")

        if skip_video == True:
            continue

        cv2.destroyWindow("Select Camera")
        roi_data_list = list(camera_data["data"].values())
        roi_lookup ={roi["position"]: roi for roi in roi_data_list}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            results = kp_model(frame)

            for result in results:
                keypoints_tensor = result.keypoints.data
                for person_idx, kp_tensor in enumerate(keypoints_tensor):
                    keypoints = []
                    feature_dict = {}

                    for i, keypoint in enumerate(kp_tensor):
                        x, y, conf = keypoint[:3].cpu().numpy()
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        if conf < 0.5:
                            # x, y = 0, 0  # when not using normalization
                        #uncoment following line when using normalization approach
                        # Start of normalization approach
                            x, y = -1, -1 
                        else:
                            x = x / 1280
                            y = y / 720
                        # End of normalization approach

                        keypoints.append((x, y,conf)) 
                        feature_dict[f"kp_{i}_x"] = x 
                        feature_dict[f"kp_{i}_y"] = y 
                        

                    # distances = calculate_distances(keypoints, connections)
                    # feature_dict.update(distances)
                    # A = (int(feature_dict['kp_0_x']), int(feature_dict['kp_0_y']))
                    # print(feature_dict)

                    #uncomment following line when not using normalization approach
                    # if len(keypoints) == 0 or all((x == 0 and y == 0) for x, y,_ in keypoints):
                    #     continue
                    # person_x = keypoints[0][0] 
                    
                    # for normalization approach
                    #start of normalization approach
                    if len(keypoints) == 0 or all((x == -1 and y == -1) for x, y,_ in keypoints):
                        continue
                    
                    # if conf > 0.5:
                    person_x = keypoints[0][0] * 1280  # Convert back to pixel coordinates
                #end of normalization approach
                    # person_x = keypoints[0][0]  # for without normalization
                    position = assign_roi_index(person_x)
                    roi_data = roi_lookup.get(position, None)
                    if roi_data is None:
                        print(f"ROI data not found for position {position}.")
                        continue
                    # print(f"ROI data for position {position}, desk: {roi_data['desk']}")
                    cv2.putText(frame, f"Desk: {roi_data['desk']}, Position: {roi_data['position']}", 
                                (int(person_x), 100 + person_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (130, 180, 0), 2)
                                # (int(600), int(50)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (130, 180, 0), 2)

                    # if person_x < 680:
                    #     position = -1
                    # elif 465 < person_x < 895:
                    #     position = 1
                    # elif 610 < person_x < 953:
                    #     position = 0
                    # else:
                    #     position = 1
                    feature_dict['position'] = position

                    ordered_frame_features = {key: feature_dict.get(key, 0.0) for key in single_frame_features}
                    # sliding_window.append(ordered_frame_features)

                    if position not in sliding_window:
                        sliding_window[position] = deque(maxlen=WINDOW_SIZE)

                    sliding_window[position].append(ordered_frame_features)

                    # Predict only if window is full
                    if len(sliding_window[position]) == WINDOW_SIZE:
                        flat_feature = {}
                        # for idx, frame_feats in enumerate(sliding_window):
                        for t, frame_feats in enumerate(sliding_window[position]):
                            for key, value in frame_feats.items():
                                flat_feature[f"{key}_t{t}"] = value

                        input_df = pd.DataFrame([flat_feature])
                        # print(input_df)

                        # input_df.to_csv("input_df_o.csv", index=True, sep='\n')

                        prediction = rf_model.predict(input_df)[0]

                        # label = "Hand in Pocket" if prediction == 1 else "No Hand in Pocket"
                        if prediction == 1:
                            cv2.putText(frame, "Hand in Pocket", (int(person_x), 50 + person_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0, 255), 2)
                            # cv2.putText(frame, "Hand in Pocket", (int(50), int(50)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0, 255), 2)
                            
                        else:
                            cv2.putText(frame, "No Hand in Pocket", (int(person_x), 50 + person_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            # cv2.putText(frame, "No Hand in Pocket", (int(50), int(50)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Temporal Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
