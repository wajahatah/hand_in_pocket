import os
import pandas as pd

# 📂 Folder containing CSVs
input_folder = "C:/wajahat/hand_in_pocket/dataset/training2/balanced/old_hp/acting"
output_folder = "C:/wajahat/hand_in_pocket/dataset/training2/balanced/old_hp/"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all CSVs
for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(input_folder, file_name)

        # Read CSV
        df = pd.read_csv(file_path)

        # Keep every second row starting from row index 0 (which is dataset row 2)
        df_new = df.iloc[::2].reset_index(drop=True)

        # Save with new name
        new_file_name = file_name.replace("_keypoints.csv", "_skiprate.csv")
        output_path = os.path.join(output_folder, new_file_name)
        df_new.to_csv(output_path, index=False)

        print(f"Processed: {file_name} → {new_file_name}")
