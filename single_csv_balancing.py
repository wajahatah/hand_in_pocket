import os
import pandas as pd
from tkinter import Tk, filedialog, Frame
from pandastable import Table

# Paths
input_folder = "C:/wajahat/hand_in_pocket/dataset/training2/ali_fp"   # folder containing original csvs
output_folder = "C:/wajahat/hand_in_pocket/dataset/training2/balanced/fp_hp" # folder to save modified csvs
os.makedirs(output_folder, exist_ok=True)

# Get list of CSV files
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

for csv_file in csv_files:
    file_path = os.path.join(input_folder, csv_file)
    print(f"\nOpening {csv_file}...\n")

    # Load CSV
    df = pd.read_csv(file_path)
    # print(df)

    root = Tk()
    root.title(f"Editing {csv_file} - Close window when done")

    frame = Frame(root)
    frame.pack(fill="both", expand=True)

    # Create table widget
    table = Table(frame, dataframe=df, showtoolbar=True, showstatusbar=True)
    table.show()

    # Start GUI loop (user edits with mouse, deletes rows via right-click menu)
    root.mainloop()

    # Get updated dataframe
    updated_df = table.model.df

    # while True:
    #     try:
    #         delete_idx = input("Enter row index(es) to delete (comma-separated), or press Enter if done: ")
    #         if delete_idx.strip() == "":
    #             break
    #         delete_idx = [int(x.strip()) for x in delete_idx.split(",")]
    #         df = df.drop(delete_idx).reset_index(drop=True)
    #         print("\nUpdated CSV:")
    #         print(df)
    #     except Exception as e:
    #         print(f"Error: {e}. Try again.")

    # Save modified CSV
    save_path = os.path.join(output_folder, csv_file)
    # df.to_csv(save_path, index=False)
    updated_df.to_csv(save_path, index=False)
    print(f"Saved updated file to {save_path}")
