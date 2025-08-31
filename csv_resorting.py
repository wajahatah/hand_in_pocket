""""resorting the csv to do the balancig of TP and FN samples of dataset"""


import pandas as pd

# --- Load CSV ---
input_csv = "C:/wajahat/hand_in_pocket/dataset/new_dataset/new_combined.csv"   # <-- replace with your CSV path
output_csv = "C:/wajahat/hand_in_pocket/dataset/new_dataset/new_combined_sorted.csv"

# Read the CSV
df = pd.read_csv(input_csv)

# Ensure numeric sorting:
# Extract numbers from 'camera' and 'video' columns
df['camera_num'] = df['camera'].str.extract(r'(\d+)').astype(int)
df['video_num'] = df['video'].str.extract(r'(\d+)').astype(int)
df['desk'] = df['desk'].astype(int)
df['frame'] = df['frame'].astype(int)

# Sort by camera_num → video_num → desk → frame
df_sorted = df.sort_values(by=['camera_num', 'video_num', 'desk', 'frame'],
                           ascending=[True, True, True, True])

# Drop helper columns if you don’t want them in output
df_sorted = df_sorted.drop(columns=['camera_num', 'video_num'])

# Save to new CSV
df_sorted.to_csv(output_csv, index=False)

print(f"Rows rearranged and saved to {output_csv}")
