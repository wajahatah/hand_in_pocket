import pandas as pd
import os
import re

input_csv = "temp_keypoint_l1_v2_norm"
df = pd.read_csv(f"C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/{input_csv}.csv")

columns = df.columns[1:-1]
col_list = df.columns.tolist()

def sorting_keys(col):
    match = re.search(r'kp_(\d+)_(x|y)_t(\d+)', col)
    if match:
        kp_index = int(match.group(1))
        axis = 0 if match.group(2) == 'x' else 1
        time_index = int(match.group(3))
        return (kp_index, axis, time_index)
    
    else:
        return (999, 999, 999)
    
sorted_columns = sorted(columns, key=sorting_keys)
sorted_columns = ['source_file'] + sorted_columns + ['hand_in_pocket']
print(sorted_columns)

df = df[sorted_columns]
df.to_csv(f"C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/{input_csv}_sorted.csv", index=False)
print(f"Sorted keypoints saved to {input_csv}_sorted.csv")