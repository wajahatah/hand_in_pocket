import os
import cv2
import pandas as pd
from collections import defaultdict

# =============================
# USER CONFIGURATION
# =============================
videos_folder = "C:/Users/LT/Downloads/combine_testbench/test_room2/FP"       # Folder containing videos
excel_path = "C:/Users/LT/Downloads/combine_testbench/Evaluations_new.xlsx" # Existing Excel (for reference)
output_excel_path = "C:/Users/LT/Downloads/combine_testbench/video_rename_log.xlsx"  # New Excel to save names

room_num = 2                     # Predefined room number
type_label = "fp"                # Either "fp" or "tp"
video_ext = ".mp4"               # Video file extension
excel_video_col = "Video_Name"   # Column in existing Excel with original video names
# =============================

# Load Excel (for reference)
df_original = pd.read_excel(excel_path)

# Keep track of desk counts
desk_counts = defaultdict(int)

# To store rename log
rename_log = []

# Iterate through all videos in the folder
for filename in os.listdir(videos_folder):
    if not filename.endswith(video_ext):
        continue

    video_path = os.path.join(videos_folder, filename)
    base_name = os.path.splitext(filename)[0]

    print(f"\n▶️ Playing video: {filename}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open {filename}")
        continue

    # Play video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video Player", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("⏹️ Video skipped by user.")
            break
    
    desk_num = input("Enter desk number for this video (or press Enter to skip): ").strip() 
    if desk_num == "": 
        print("⏭️ Skipped renaming for this video.") 
        continue

    cap.release()
    cv2.destroyAllWindows()

    # Take user input for desk number
    # desk_num = input("Enter desk number for this video (or press Enter to skip): ").strip()
    # if desk_num == "":
    #     print("⏭️ Skipped renaming for this video.")
    #     continue

    # Update desk count for numbering
    desk_counts[desk_num] += 1
    count = desk_counts[desk_num]

    # Generate new name
    new_name = f"r{room_num}_d{desk_num}_{type_label}{count}"
    new_path = os.path.join(videos_folder, new_name + video_ext)

    # Rename video
    try:
        os.rename(video_path, new_path)
        print(f"✅ Renamed: {filename} → {new_name}{video_ext}")
    except Exception as e:
        print(f"❌ Error renaming {filename}: {e}")
        continue

    # Find corresponding entry in Excel (if any)
    match_row = df_original[df_original[excel_video_col].astype(str) == base_name]
    if not match_row.empty:
        old_name = match_row[excel_video_col].values[0]
    else:
        old_name = base_name  # fallback if not found

    # Add to rename log
    rename_log.append({
        "Old_Name": old_name,
        "New_Name": new_name
    })

# Save the new Excel with rename log
if rename_log:
    df_log = pd.DataFrame(rename_log)
    df_log.to_excel(output_excel_path, index=False)
    print(f"\n📘 Rename log saved to: {output_excel_path}")
else:
    print("\n⚠️ No videos were renamed.")
