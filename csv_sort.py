import pandas as pd
import os

# def sort_csv(input_csv, output_csv):
def sort_csv(input_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Sort the DataFrame by 'hand_in_pocket' and 'source_file'
    sorted_df = df.sort_values(by=['desk_no', 'frame'], ascending=[True, True])

    # Save the sorted DataFrame to a new CSV file
    sorted_df.to_csv(input_csv, index=False)
    print(f"✅ Sorted CSV saved as: {input_csv}")

input_csv = "C:/wajahat/hand_in_pocket/dataset/training/c5_v1.csv"
# output_csv = "C:/wajahat/hand_in_pocket/dataset/training/c5_v1.csv"
sort_csv(input_csv)