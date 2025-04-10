import pandas as pd
import os

input_csv_path = "C:/wajahat/hand_in_pocket/dataset/training/c2_v3.csv"
keypoint_output_dir = "C:/wajahat/hand_in_pocket/dataset/split_keypoint"
distance_output_dir = "C:/wajahat/hand_in_pocket/dataset/split_distance"
os.makedirs(keypoint_output_dir,exist_ok=True)  # Create output directory if it doesn't exist
os.makedirs(distance_output_dir,exist_ok=True)  # Create output directory if it doesn't exist

df = pd.read_csv(input_csv_path)

base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
keypoint_columns = ['frame','person_idx','desk_no','position'] 
keypoint_columns += [col for col in df.columns if col.startswith('kp_') and  ('_x' in col or '_y' in col)]
keypoint_columns.append('hand_in_pocket')

distance_columns = ['frame','person_idx','desk_no','position']
distance_columns += [col for col in df.columns if col.startswith('distance(')]
distance_columns.append('hand_in_pocket')

df_keypoints = df[keypoint_columns]
df_distances = df[distance_columns]

df_keypoints.to_csv(os.path.join(keypoint_output_dir, f"{base_name}_keypoints.csv"), index=False)
df_distances.to_csv(os.path.join(distance_output_dir, f"{base_name}_distances.csv"), index=False)

print("Keypoints CSV saved")