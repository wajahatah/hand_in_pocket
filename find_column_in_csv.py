import pandas as pd
import os
import glob

# Path where your CSVs are stored
input_folder = "C:/wajahat/hand_in_pocket/dataset/training2/balanced/old_hp"
target_column = "hand_in_pocket"

csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

for file in csv_files:
    try:
        df = pd.read_csv(file)

        # Check if column exists
        if target_column not in df.columns:
            print(f"❌ Column '{target_column}' is MISSING in file: {os.path.basename(file)}")
            continue

        col_data = df[target_column]

        # Check if column is empty
        if col_data.dropna().empty:
            print(f"⚠️ Column '{target_column}' is EMPTY in file: {os.path.basename(file)}")
            continue

        # Unique values in the column
        unique_vals = col_data.dropna().unique()

        # Check if only contains 0/1
        if not set(unique_vals).issubset({0, 1}):
            print(f"⚠️ Column '{target_column}' in {os.path.basename(file)} contains unexpected values: {unique_vals}")

    except Exception as e:
        print(f"⚠️ Error reading {os.path.basename(file)}: {e}")
