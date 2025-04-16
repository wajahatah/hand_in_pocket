import pandas as pd
import os
import glob

input_folder = "C:/wajahat/hand_in_pocket/dataset/training"
output_dir = "C:/wajahat/hand_in_pocket/dataset/training/combined"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
output_file = 'temp_tp1.csv'
output_file = os.path.join(output_dir, output_file)

window_size = 5  # Size of the rolling window
stride = 1
target_column = 'hand_in_pocket'

columns_to_drop = ['frame', 'person_idx', 'desk_no']

all_temporal_rows= []
feature_cols = None

csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

for file in csv_files:
    
    df = pd.read_csv(file)

    if df.empty or len(df) < window_size:
        print(f"Skipping empty or too short file: {file}")
        continue

    drop_cols = columns_to_drop + [col for col in df.columns if '_conf' in col]
    df.drop(columns=[ col for col in drop_cols if col in df.columns], inplace=True)

    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_column]
        # print(f"Feature columns: {feature_cols}")
        assert len(feature_cols) * window_size == 150, "Feature columns length match"

    file_name = os.path.basename(file)

    for i in range(0, len(df)-window_size+1, stride):
        window = df.iloc[i:i+window_size]

        if len(window) < window_size:
            print(f"Skipping incomplete window at index {i} for {file_name}")
            continue

        feature = window[feature_cols].values.flatten()

        label_counts = window[target_column].value_counts()
        # label = 1 if label_counts.get(1, 0) > label_counts.get(0, 0) else 0
        label = 1 if label_counts.get(1, 0) >= 1 else 0

        all_temporal_rows.append([file_name] + feature.tolist() + [label])


    
temporal_features_cols = [f"{col}_t{t}" for t in range(window_size) for col in feature_cols]

output_columns = ['source_file'] + temporal_features_cols + [target_column]

temporal_df = pd.DataFrame(all_temporal_rows, columns=output_columns)
temporal_df.to_csv(os.path.join(output_dir, output_file), index=False)

print("✅ Temporal features CSV saved")