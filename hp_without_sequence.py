import pandas as pd

input_csv = "C:/wajahat/hand_in_pocket/dataset/training2/new_combined_temp_balanced_norm_reg.csv"
output_csv = "C:/wajahat/hand_in_pocket/dataset/training2/new_combined_temp_balanced_norm_without_seq.csv"

df = pd.read_csv(input_csv)

df_cleaned = df[df["hand_in_pocket"].isin([0,1])]

df_cleaned.to_csv(output_csv, index=False)

print(f"Cleaned CSV saved as {output_csv}, original rows: {len(df)}, cleaned rows: {len(df_cleaned)}")
