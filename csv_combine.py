import pandas as pd
import os
import glob

# Path to your folder with the 17 CSV files
input_folder = "C:/wajahat/hand_in_pocket/dataset/split_keypoint"
output_dir = "C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
output_file = 'combined_2.csv'
output_file = os.path.join(output_dir, output_file)

# Load and combine all CSVs
all_files = glob.glob(os.path.join(input_folder, '*.csv'))

df_list = []
for file in all_files:
    df = pd.read_csv(file)

    # Drop unwanted columns
    columns_to_drop = ['frame', 'person_idx', 'desk_no']
    # columns_to_drop = ['frame', 'person_idx']
    columns_to_drop += [col for col in df.columns if '_conf' in col]

    df.drop(columns=columns_to_drop, inplace=True)

    # df['source_file'] = os.path.basename(file)
    df.insert(0, 'source_file', os.path.basename(file))  # Add source file name as the first column

    df_list.append(df)

# Combine all dataframes
combined_df = pd.concat(df_list, ignore_index=True)

# Save to CSV
combined_df.to_csv(output_file, index=False)

print(f"✅ Combined CSV saved as: {output_file}")
print(f"🎯 Target column: 'hand_in_pocket'")
