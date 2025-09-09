import pandas as pd
import os
import glob

# input_folder = "C:/wajahat/hand_in_pocket/dataset/training"
input_folder = "C:/wajahat/hand_in_pocket/dataset/split_keypoint"
# output_dir = "C:/wajahat/hand_in_pocket/dataset/training/combined"
output_dir = "C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
output_file = 'temp_distance_l1_v2.csv'
output_file = os.path.join(output_dir, output_file)

window_size = 5  # Size of the rolling window
stride = 1
target_column = 'hand_in_pocket'

# columns_to_drop = ['frame', 'person_idx', 'desk_no']
columns_to_drop = ['person_idx']
meta_columns = ['frame','desk_no']
special_column = 'position'


all_temporal_rows= []
feature_cols = None

csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

for file in csv_files:
    
    df = pd.read_csv(file)

    if df.empty or len(df) < window_size:
        print(f"Skipping empty or too short file: {file}")
        continue

    # drop_cols = columns_to_drop + [col for col in df.columns if '_conf' in col]
    # df.drop(columns=[ col for col in drop_cols if col in df.columns], inplace=True)
    drop_cols = [col for col in df.columns if '_conf' in col or col.startswith("distance(")]
    df.drop(columns=drop_cols + columns_to_drop, inplace=True, errors="ignore")

    if feature_cols is None:
        feature_cols = [col for col in df.columns 
                        # if col != target_column
                        if col not in meta_columns + [special_column, target_column]]
        # print(f"Feature columns: {feature_cols}")
        # assert len(feature_cols) * window_size == 150, "Feature columns length match" # for combined csvs
        assert len(feature_cols) * window_size > 0 # == 105, "Feature columns length match" # for split csvs

    file_name = os.path.basename(file)

    # group by desk_no 

    for desk_no, desk_group in df.groupby("desk_no"):
        desk_group = desk_group.reset_index(drop=True)

        if len(desk_group) < window_size:
            print(f"skipping desk {desk_no} in {file_name} due to less rows")
            continue

        for i in range(0, len(desk_group) - window_size + 1, stride):
            window = desk_group.iloc[i:i + window_size]

            if len(window) < window_size:
                continue

            feature = window[feature_cols].values.flatten()

            frame_val = window[meta_columns[0]].iloc[0]
            desk_val = window[meta_columns[1]].iloc[0]

            position_val = window[special_column].iloc[0] if special_column in window.columns else None

            labels_counts = window[target_column].value_counts()
            # label = 1 if labels_counts.get(1,0) >= 1 else 0  # for the logic if there is 1 in the window, leabel is 1
            one_count = labels_counts.get(1,0)  # for new regresion logic
            label = one_count / window_size

            all_temporal_rows.append([file_name, frame_val, desk_val] + feature.tolist() + [position_val, label])

temporal_features_cols = [
    f"{col}_t{t}" for t in range(window_size) for col in feature_cols
]

output_columns = ['source_file'] + meta_columns + temporal_features_cols + [special_column, target_column]

temporal_df = pd.DataFrame(all_temporal_rows, columns=output_columns)
temporal_df.to_csv(output_file, index=False)



#     for i in range(0, len(df)-window_size+1, stride):
#         window = df.iloc[i:i+window_size]

#         if len(window) < window_size:
#             print(f"Skipping incomplete window at index {i} for {file_name}")
#             continue

#         feature = window[feature_cols].values.flatten()

#         label_counts = window[target_column].value_counts()
#         # label = 1 if label_counts.get(1, 0) > label_counts.get(0, 0) else 0
#         label = 1 if label_counts.get(1, 0) >= 1 else 0

#         all_temporal_rows.append([file_name] + feature.tolist() + [label])


    
# temporal_features_cols = [f"{col}_t{t}" for t in range(window_size) for col in feature_cols]

# output_columns = ['source_file'] + temporal_features_cols + [target_column]

# temporal_df = pd.DataFrame(all_temporal_rows, columns=output_columns)
# temporal_df.to_csv(os.path.join(output_dir, output_file), index=False)

print("✅ Temporal features CSV saved")