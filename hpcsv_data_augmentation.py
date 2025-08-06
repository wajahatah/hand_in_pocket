import pandas as pd
import numpy as np
import random

# === Settings ===
csv_path = "C:/wajahat/hand_in_pocket/dataset/without_kp/no_hp_annotations2.csv"       # Change this to your file path
output_path = "C:/wajahat/hand_in_pocket/dataset/without_kp/no_hp_augmented_annotations2.csv"
augment_times = 1
randomize_per_column = True       # Set to False to use same number for all 100 columns

# === Load Data ===
df = pd.read_csv(csv_path)

# === Identify x/y columns ===
xy_columns = [col for col in df.columns if 'x' in col.lower() or 'y' in col.lower()]
non_xy_columns = [col for col in df.columns if col not in xy_columns]

# === Prepare for augmentation ===
augmented_rows = []

for idx, row in df.iterrows():
    original_row = row.copy()
    for col in xy_columns:
        if original_row[col] != 0:
            original_row[col] = int(original_row[col])
    original_row['original'] = 1  # Mark original row
    augmented_rows.append(original_row)

    for _ in range(augment_times):
        new_row = row.copy()

        if randomize_per_column:
            # Different noise per column
            for col in xy_columns:
                if new_row[col] != 0:                    
                    original_val = int(row[col])
                    noise = random.randint(-4, 4)
                    new_row[col] = original_val + noise
                else:
                    new_row[col] = 0  # If original was 0, keep it 0
        else:
            # Same noise for all
            noise = random.randint(-4, 4)
            for col in xy_columns:
                if new_row[col] != 0:
                    original_val = int(row[col])
                    new_row[col] = original_val + noise
                else:
                    new_row[col] = 0  # If original was 0, keep it 0

        new_row['original'] = 0  # Mark augmented row
        augmented_rows.append(new_row)

# === Final DataFrame ===
augmented_df = pd.DataFrame(augmented_rows)

# === Save to CSV ===
augmented_df.to_csv(output_path, index=False)

print(f"Augmentation completed. Output saved to: {output_path}")
