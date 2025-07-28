import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from collections import deque
from ultralytics import YOLO
import time
import statistics as stats

# ========== MLP Model ==========
class MLP(nn.Module):
    def __init__(self, input_size=104, hidden_size=64):
        super(MLP, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(104, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_size, hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_size // 2, 1),
        #     nn.Sigmoid()
        # )

        # # for c1 model architecture
        # self.net = nn.Sequential(
        #     nn.Linear(104, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(16, 1),
        #     nn.Sigmoid()
        # )

        # for c2 model architecture
        # self.net = nn.Sequential(
        #     nn.Linear(104, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(32, 1),
        #     nn.Sigmoid()
        # )

        # for c3 model architecture
        # self.net = nn.Sequential(
        #     nn.Linear(104, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 1),
        #     nn.Sigmoid()
        # )

        # for c4 model architecture
        self.net = nn.Sequential(
            nn.Linear(104, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ========== ROI Assignment ==========
roi_data_list = []
def assign_roi_index(x):
    for roi in roi_data_list:
        if roi["xmin"] <= x < roi["xmax"]:
            return roi["position"]

# ========== Load Model ==========
def load_mlp_model(weights_path, device):
    model = MLP()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


mlp_times = []
video_num = 0

# ========== Main Inference ==========
if __name__ == "__main__":
    kp_model = YOLO("C:/wajahat/hand_in_pocket/bestv8-1.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model = load_mlp_model("rf_models/mlp_temp_norm_regrouped_pos_gen-c4.pt", device)

    # input_dir = "C:/Users/LAMBDA THETA/Videos"
    # input_dir = "C:/Users/LAMBDA THETA/Downloads/july_27/fp/Hands In Pocket"
    # input_dir = "F:/Wajahat/looking_around_panic/may_8/Hands In Pocket1/TP"
    json_path = "qiyas_multicam.camera_final.json"
    WINDOW_SIZE = 5
    frame_num = 0

    # video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    # if not video_files:
    #     print("No videos found.")
    #     exit()

    # for video_file in video_files:
        # video_path = os.path.join(input_dir, video_file)
    while True:
        video_file = video_path = "C:/Users/LAMBDA THETA/Downloads/july_27/fp/Hands In Pocket/1753613382.466142.mp4"
        print(f"Processing: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error loading video.")
            continue

        video_file_name = video_file

        # Get camera ID
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Select Camera", frame)
        cv2.waitKey(1)

        with open(json_path, 'r') as f:
            camera_config = json.load(f)

        skip_video = False
        while True:
            cam_id = input("Enter camera ID: ")
            if cam_id.lower() == 's':
                skip_video = True
                cap.release()
                cv2.destroyWindow("Select Camera")
                break
            cam_key = f"camera_{cam_id}"
            camera_data = next((cam for cam in camera_config if cam["_id"] == cam_key), None)
            if camera_data:
                break
            print("Invalid camera ID. Try again.")

        if skip_video:
            continue

        cv2.destroyWindow("Select Camera")
        roi_data_list = list(camera_data["data"].values())
        roi_lookup = {roi["position"]: roi for roi in roi_data_list}
        sliding_window = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            results = kp_model(frame)

            batch_input_vector = []
            for result in results:
                if not hasattr(result, 'keypoints') or result.keypoints is None:
                    continue

                keypoints_tensor = result.keypoints.data
                frame_num += 1

                for person_idx, kp_tensor in enumerate(keypoints_tensor):
                    keypoints = []
                    feature_dict = {}

                    for i, keypoint in enumerate(kp_tensor):
                        x, y, conf = keypoint[:3].cpu().numpy()
                        print("type of x:", type(x))
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        if conf < 0.5:
                            x, y = -1, -1
                        else:
                            x = x / 1280
                            y = y / 720
                        feature_dict[f"kp_{i}_x"] = x
                        feature_dict[f"kp_{i}_y"] = y
                        keypoints.append((x, y))

                    if len(keypoints) == 0 or all((x == -1 and y == -1) for x, y in keypoints):
                        continue

                    person_x = keypoints[0][0] * 1280
                    position = assign_roi_index(person_x)
                    roi_data = roi_lookup.get(position)
                    if not roi_data:
                        continue

                    feature_dict['position'] = position
                    if position not in sliding_window:
                        sliding_window[position] = deque(maxlen=WINDOW_SIZE)
                    f = sliding_window[position].append(feature_dict)

                    # print(f"feature_dict: {feature_dict}")
                    # print(f"array: {f}")
                    

                    if len(sliding_window[position]) == WINDOW_SIZE:
                        flat_feature = {}
                        for i in range(10):
                            for axis in ['x', 'y']:
                                for t in range(WINDOW_SIZE):
                                    flat_feature[f"kp_{i}_{axis}_t{t}"] = sliding_window[position][t][f"kp_{i}_{axis}"]
                        # for t in range(WINDOW_SIZE):
                        #     flat_feature[f"position_t{t}"] = sliding_window[position][t]["position"]
                        pos_list = roi_data.get("position_list", [0, 0, 0, 0])
                        flat_feature["position_a"] = pos_list[0]
                        flat_feature["position_b"] = pos_list[1]
                        flat_feature["position_c"] = pos_list[2]
                        flat_feature["position_d"] = pos_list[3]

                        ordered_columns = [f"kp_{i}_{axis}_t{t}" for i in range(10) for axis in ['x', 'y'] for t in range(WINDOW_SIZE)]
                        ordered_columns.extend(["position_a", "position_b", "position_c", "position_d"])

                        # print(f"Ordered columns: {ordered_columns}")
                        # ordered_columns += [f"position_t{t}" for t in range(WINDOW_SIZE)]
                        
                        # for single frame inference
                        input_tensor = torch.tensor([[flat_feature[col] for col in ordered_columns]], dtype=torch.float32).to(device)
                        # print(f"Position: {position}")
                        print(f"Position: {roi_data['desk']}")
                        print(f"frame num: {frame_num}")
                        print(f"input_tensor: {input_tensor}, shape: {input_tensor.shape}")

                        # frame_num += 1
                        with torch.no_grad():
                            start = time.time()
                            # print(f"start_time: {start}")
                            prob = mlp_model(input_tensor)
                            print(f"prob: {prob}")
                            end = time.time()
                            # print(f"end_time: {end}")
                            mlp_times.append(end - start)
                            # print(f"Prediction time: {(end - start)*1000:.4f} seconds")
                            # print(f"[Person {person_idx} | Pos {position}] Start: {start:.6f}, End: {end:.6f}, Time: {(end - start)*1000:.4f} ms")
                            prediction = 1 if prob >= 0.5 else 0

                            if prediction == 1:
                                if video_file == video_file_name:
                                    with open("prediction.csv", "a", newline='') as f:
                                        f.write(f"{video_file}, {prediction}, Hand in Pocket \n")
                                    video_file_name = f"{video_file}_done"


                        # for batch inference of mlp model
                        # start here 
                        # input_vector = [flat_feature[col] for col in ordered_columns]

                        # batch_tensor = torch.tensor([input_vector for _ in range(10)], dtype=torch.float32).to(device)
                        # batch_input_vector.append((person_idx, position, person_x, roi_data, input_vector))
                        # print("##################################################")
                        # print(f"Batch Tensor: {batch_tensor}")

                        # if len(batch_input_vector) >1:
                            # batch_tensor = torch.tensor( 
                            #     [vec for (_,_,_,_, vec) in batch_input_vector],
                            #     dtype= torch.float32).to('cuda')
                            # import random
                            # input_vector = [vec for (_,_,_,_,vec) in batch_input_vector]

                    

                        # batch_tensor = torch.tensor([input_vector] * 32, dtype=torch.float32).to(device)    
                        # print(f"batch tensor: {batch_tensor}")                        

                        # with torch.no_grad():
                        #     start = time.time()
                        #     probs = mlp_model(batch_tensor)
                        #     print(f"probs: {probs}")
                        #     end = time.time()
                        #     mlp_times.append(end - start)
                        #     # print(f"[Person {person_idx} | Pos {position}] Start: {start:.6f}, End: {end:.6f}, Time: {(end - start)*1000:.4f} ms")
                        #     print(f"[MLP] Batch inference on {position} persons. Time: {(end - start) * 1000:.3f} ms")

                        
                        # for i in range(10):
                            # for i, (person_idx, position, person_x, roi_data, _) in enumerate(batch_input_vector):
                            # for i, prob_tensor in enumerate(probs):
                            #     # prob = probs[i].item()
                            #     prob = prob_tensor.item()
                                
                            #     prediction = 1 if prob >= 0.5 else 0
                            #     print(f"prectiction: {prediction}")
                            #     label = "Hand in Pocket" if prediction else "No Hand in Pocket"
                            #     color = (0, 0, 255) if prediction else (0, 255, 0)

                            #     cv2.putText(frame, f"{label} ({prob:.2f})", (int(person_x), 50 + person_idx * 30),
                            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            #     cv2.putText(frame, f"Desk: {roi_data['desk']}, Pos: {roi_data['position']}",
                            #                 (int(person_x), 100 + person_idx * 30),
                            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 180, 0), 2)
                                
            # batch_input_vector = []
                        # end here

                        # uncomment below lines for single frame inference
                        label = "Hand in Pocket" if prediction else "No Hand in Pocket"
                        color = (0, 0, 255) if prediction else (0, 255, 0)
                        # cv2.putText(frame, f"{label} ({prob:.2f})", (int(person_x), 50 + person_idx * 30),
                                    # cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"Desk: {roi_data['desk']}, Pos: {roi_data['position']}",
                                    (int(person_x), 100 + person_idx * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 180, 0), 2)
                        

                        # uncomment below lines to stop the video on prediction 
                        # cv2.imshow("GRU Inference", frame)
                        # if prediction == 1:
                        #     cv2.waitKey(1)
                        # else:
                        #     if cv2.waitKey(1) & 0xFF == ord('q'):
                        #         break

            cv2.imshow("MLP Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_num += 1
        print(f"video: {video_num}")
        cap.release()


    # Print MLP inference statistics
        if mlp_times:
            avg_time = sum(mlp_times) / len(mlp_times)
            median_time = stats.median(mlp_times)

            nonzero = [t for t in mlp_times if t > 0]
            if nonzero:
                try:
                    # mode_time = stats.mode(mlp_times)
                    mode_time = stats.mode(nonzero)
                except stats.StatisticsError:
                    mode_time = "No unique mode"

            
            # print(f"Average MLP inference time: {avg_time:.4f} seconds")
            print(f"[MLP] Average inference time: {avg_time*1000:.3f} ms over {len(mlp_times)} samples")
            print(f"Max: {max(mlp_times)*1000:.3f} ms, Min: {min(mlp_times)*1000:.3f} ms")
            print(f"Median : {median_time * 1000:.3f} ms")
            print(f"Mode   : {mode_time if isinstance(mode_time, str) else f'{mode_time * 1000:.3f} ms'}")


    cv2.destroyAllWindows()
