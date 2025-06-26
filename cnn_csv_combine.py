import pandas as pd
import os

def combine_csvs(folder_path, output_csv):
    all_dfs = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)

            # Load CSV
            df = pd.read_csv(file_path)

            # Drop 'person_id' column if it exists
            if 'person_idx' in df.columns:
                df = df.drop(columns=['person_idx'])

            # Modify the file name: remove '_keypoints', add '.csv'
            source_name = file_name.replace('_keypoints', '')

            # Add 'source_file' column at the start
            df.insert(0, 'source_file', source_name)

            all_dfs.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save to output CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"✅ Combined CSV saved to: {output_csv}")

# ======== CONFIG ========
input_folder = "C:/wajahat/hand_in_pocket/dataset/split_keypoint"       # e.g., "C:/data/keypoints_csvs"
output_file = "C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/cnn_combine.csv"        # e.g., "C:/data/combined.csv"

# ======== RUN ========
combine_csvs(input_folder, output_file)
