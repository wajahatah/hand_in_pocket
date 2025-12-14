import pandas as pd
import json
import re

# === File paths ===
csv = "tp_s1_w1"
# input_csv = f"C:/wajahat/hand_in_pocket/dataset/training2/window4/seq2/{csv}.csv"
input_csv = f"C:/wajahat/hand_in_pocket/dataset/training3/{csv}_combine.csv"
output_dir = "C:/wajahat/hand_in_pocket/dataset/training3/"
csv_name = f'{csv}_pos.csv'
output_csv = f"{output_dir}/{csv_name}"
json_file = "C:/wajahat/hand_in_pocket/qiyas_multicam.camera_final.json"
# json_file = "qiyas_multicam_2.camera.json"

# === Load data ===
df = pd.read_csv(input_csv)
with open(json_file, 'r') as f:
    camera_data = json.load(f)

# === Build camera map from JSON ===
camera_map = {}
for cam in camera_data:
    if '_id' in cam:
        match = re.search(r'camera_(\d+)', cam['_id'])
        if match:
            cam_id = int(match.group(1))
            camera_map[cam_id] = cam

# === Process rows for temporal feature ===
# processed_rows = []

# for idx, row in df.iterrows():
#     try:
#         # Extract and validate position values
#         pos_vals = [row[f'position_t{i}'] for i in range(5)]
#         if len(set(pos_vals)) != 1:
#             # print(f"Row {idx} skipped — Position values not uniform: {pos_vals}")
#             continue  # Skip if position values differ

#         position_val = pos_vals[0]

#         # Extract camera number from source_file
#         source_file = row['source_file']
#         match = re.search(r'c(\d+)_v\d+', source_file)
#         if not match:
#             continue  # Skip if camera number not found

#         cam_id = int(match.group(1))
#         cam_info = camera_map.get(cam_id)
#         if not cam_info:
#             continue

#         # Look for matching position in camera JSON data
#         matched_entry = None
#         for entry in cam_info['data'].values():
#             if entry.get('position') == position_val:
#                 matched_entry = entry
#                 break

#         if not matched_entry or 'position_list' not in matched_entry:
#             continue

#         position_list = matched_entry['position_list']
#         if len(position_list) != 4:
#             continue  # Skip if not exactly 4 values

#         # Build new row with replaced columns
#         new_row = row.drop(labels=[f'position_t{i}' for i in range(5)]).to_dict()
#         new_row['position_a'], new_row['position_b'], new_row['position_c'], new_row['position_d'] = position_list
#         processed_rows.append(new_row)

#     except Exception as e:
#         print(f"⚠️ Skipping row {idx} due to error: {e}")


# === processing rows for single position ===

processed_rows = []

for idx, row in df.iterrows():
    try:
        position_val = row['position']

        source_file = row['source_file']
        match = re.search(r'c(\d+)_v\d+', source_file)
        # print(f"match {match}")
        if not match:
            # print(f"match not found {match}")
            continue

        cam_id = int(match.group(1))
        cam_info = camera_map.get(cam_id)
        # print("cam_info", cam_info['data'].values())
        if not cam_info:
            # print(f"cam info not found {cam_id}")
            continue

        matched_entry = None
        for entry in cam_info['data'].values():
            if entry.get('position') == position_val:
                matched_entry = entry
                break

        if not matched_entry or 'position_list' not in matched_entry:
            # print(f"matched entry not found {matched_entry}")
            continue

        position_list = matched_entry['position_list']
        if len(position_list) != 4:
            # print(f"position list not found {position_list}")
            continue

        new_row = row.drop(labels=['position']).to_dict()
        new_row['position_a'], new_row['position_b'], new_row['position_c'], new_row['position_d'] = position_list
        processed_rows.append(new_row)
        # print("processed_rows: ",processed_rows)

    except Exception as e:
        print(f"⚠️ Skipping row {idx} due to error: {e}")

# === Final DataFrame and column ordering ===
if processed_rows:
    new_df = pd.DataFrame(processed_rows)

    new_df["camera"] = new_df["source_file"].str.extract(r'^(c\d+)_')
    new_df["video"] = new_df["source_file"].str.extract(r'_(v\d+)')

    # Drop source_file
    if "source_file" in new_df.columns:
        new_df = new_df.drop(columns=["source_file"])

    col_order = ["camera", "video"] + [c for c in new_df.columns if c not in ["camera", "video"]]
    new_df = new_df[col_order]

    # Reorder: insert position_a-d before hand_in_pocket
    if 'hand_in_pocket' in new_df.columns:
        cols = list(new_df.columns)
        insert_at = cols.index('hand_in_pocket')
        for col in ['position_a', 'position_b', 'position_c', 'position_d']:
            if col in cols:
                cols.remove(col)
        cols = cols[:insert_at] + ['position_a', 'position_b', 'position_c', 'position_d'] + cols[insert_at:]
        new_df = new_df[cols]

    new_df.to_csv(output_csv, index=False)
    print(f"\n✅ Output saved to: {output_csv}")
else:
    print("⚠️ No valid rows processed.")
