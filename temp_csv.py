import pandas as pd
import os

combined_csv_path = "C:/wajahat/hand_in_pocket/dataset/training/combined/combined_2.csv"

df = pd.read_csv(combined_csv_path)

window_size = 5  # Size of the rolling window
stride = 1
target_column = 'hand_in_pocket'

feature_cols = [col for col in df.columns if col not in [target_column, 'source_file']]
print(f"Feature columns: {feature_cols}")
assert len(feature_cols) * window_size == 150, "Feature columns length match"
temporal_features = []

for sourse_name, group in df.groupby('source_file'):

    group = group.reset_index(drop=True)  # Reset index for each group

    for i in range(0, len(group)-window_size+1, stride):
        window = group.iloc[i:i+window_size]

        if len(window) < window_size:
            print(f"Skipping incomplete window at index {i} for {sourse_name}")
            continue

        feature = window[feature_cols].values.flatten()

        label_counts = window[target_column].value_counts()
        # label = 1 if label_counts.get(1, 0) > label_counts.get(0, 0) else 0
        label = 1 if label_counts.get(1, 0) >= 3 else 0

        temporal_features.append([sourse_name] + feature.tolist() + [label])

temporal_features_cols =[ f"{col}_t{t}" for t in range(window_size) for col in feature_cols]
# for t in feature_cols:
#     for col in feature_cols:
#         temporal_features_cols.append(f"{col}_t{t}")

output_columns = ['source_file'] + temporal_features_cols + [target_column]

print(f"Output columns: {len(output_columns)}")
assert len(output_columns) == 152, "Output columns length match"

temporal_df = pd.DataFrame(temporal_features, columns=output_columns)
output_dir = "C:/wajahat/hand_in_pocket/dataset/training/combined"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
temporal_df.to_csv(os.path.join(output_dir, 'temporal_csv.csv'), index=False)
print("✅ Temporal features CSV saved")