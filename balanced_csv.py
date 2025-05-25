#abdul malik approach to balance the data
# input_csv = "C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/temp_keypoint_l1_v2_norm_sorted.csv"
# output_csv = "C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/temp_kp_norm_sorted_balanced-1.csv"

import pandas as pd
import random

# Load your CSV
input_csv = "C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/temp_keypoint_l1_v2_norm_sorted.csv"
df = pd.read_csv(input_csv).reset_index(drop=True)

# Target row counts
target_0 = 30000  # Total rows where hand_in_pocket == 0
target_1 = 6000  # Total rows where hand_in_pocket == 1
block_size = 5   # Number of rows to check per block

# Output containers
selected_blocks = []
count_0 = 0
count_1 = 0
used_indices = set()
max_attempts = 100000
attempts = 0

# Start the selection loop
while (count_0 < target_0 or count_1 < target_1) and attempts < max_attempts:
    attempts += 1
    start_idx = random.randint(0, len(df) - block_size)

    # Avoid re-using the same starting index
    if start_idx in used_indices:
        continue
    used_indices.add(start_idx)

    block = df.iloc[start_idx:start_idx + block_size]
    label = block.iloc[0]['hand_in_pocket']
    same_file = block['source_file'].nunique() == 1

    if same_file:
        if label == 0 and count_0 + block_size <= target_0:
            selected_blocks.append(block)
            count_0 += block_size
        elif label == 1 and count_1 + block_size <= target_1:
            selected_blocks.append(block)
            count_1 += block_size

# Combine and save
if selected_blocks:
    result_df = pd.concat(selected_blocks, ignore_index=True)
    output_csv = "C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/temp_kp_norm_sorted_balanced-6.csv"
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Saved to '{output_csv}' with {count_0} rows (hand_in_pocket=0) and {count_1} rows (hand_in_pocket=1)")
else:
    print("⚠️ No valid blocks were found.")
