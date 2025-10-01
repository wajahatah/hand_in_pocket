import pandas as pd
import os
import shutil

csv_folder = ""
video_folder = ""
output_folder = ""

os.makedirs(output_folder, exist_ok=True)

for excel_file in os.listdir(csv_folder):
    if excel_file.endswith(".xlsx") or excel_file.endswith(".xls"):
        excel_path = os.path.join(csv_folder, excel_file)

        excel_file = os.path.splitext(excel_file)[0]
        excel_output = os.path.join(output_folder, excel_file)
        tp_folder = os.path.join(excel_output, "TP")
        fp_folder = os.path.join(excel_output, "FP")
        os.makedirs(tp_folder, exist_ok=True)
        os.makedirs(fp_folder, exist_ok=True)

        df = pd.read_excel(excel_path, sheet_name=f"{excel_file.split('_HP')[0]}")

        count = 0
        total_count = 0

        for _, row in df.iterrows():
            video_name = str(row["video"])
            tp_val = int(row["TP"])
            fp_val = int(row["FP"])

            src_path = os.path.join(video_folder, video_name)
            if not os.path.exists(src_path):
                print("video not found", video_name)
                continue

            if tp_val == 1 and fp_val == 0:
                dst_path = os.path.join(tp_folder, video_name)
                count +=1
            elif fp_val == 1 and tp_val == 0:
                dst_path = os.path.join(fp_folder, video_name)
                count +=1
            else:
                print("skipping video found in both", video_name)
                continue

            shutil.copy2(src_path, dst_path)

            total_count += count
            print(f"total videos for {excel_file} copied is {total_count}")