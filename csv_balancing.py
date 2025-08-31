import pandas as pd

# --- INPUTS ---
input_csv = "C:/wajahat/hand_in_pocket/dataset/new_dataset/new_combined_sorted.csv"
output_csv = "C:/wajahat/hand_in_pocket/dataset/new_dataset/new_combined_sorted_balanced2.csv"
# videos_to_process = None  # Example: [1, 3, 5] for v1,v3,v5 OR None for all
video_filter = {
    "c1": ["v1", "v2"],
    "c2": ["v1", "v2", "v3"],
    "c3": ["v1"],
    "c4": ["v1", "v2"],
    "c5": ["v1", "v2"],
    "c6": ["v1", "v2","v3"],
    "c7": ["v1", "v2"],
    "c8": ["v1"],
    "c9": ["v1"],
    "c10": ["v1"],
    # leave empty {} to process everything
}

# --- Read CSV ---
df = pd.read_csv(input_csv)

# Extract numeric IDs for sorting and grouping
df['camera_num'] = df['camera'].str.extract(r'(\d+)').astype(int)
df['video_num'] = df['video'].str.extract(r'(\d+)').astype(int)
df['desk'] = df['desk'].astype(int)
df['frame'] = df['frame'].astype(int)

# Sort by camera, video, desk, frame
df = df.sort_values(by=['camera_num', 'video_num', 'desk', 'frame'])

def find_one_segments(series):
    """Return list of (start_index, end_index) of consecutive 1's."""
    segments = []
    in_segment = False
    start_idx = None
    for i, val in enumerate(series):
        if val == 1 and not in_segment:
            in_segment = True
            start_idx = i
        elif val == 0 and in_segment:
            in_segment = False
            segments.append((start_idx, i-1))
    if in_segment:
        segments.append((start_idx, len(series)-1))
    return segments

balanced_rows = []

# Group by camera, video, desk
for (cam, vid, desk), group in df.groupby(['camera', 'video', 'desk']):
    # Skip if not in filter
    # if video_filter and (cam not in video_filter or vid not in video_filter[cam]):
    #     continue

    group = group.reset_index(drop=True)
    
    needs_processing = (
        not video_filter 
        or (cam in video_filter and vid in video_filter[cam])
    )

    if not needs_processing:
        balanced_rows.append(group)
        continue
    
    one_indices = group[group['hand_in_pocket'] == 1].index.tolist()

    if len(one_indices) == 0:
        # No 1's: take at most 30 rows
        print(f"No '1' found for {cam}, {vid}, desk {desk} — adding up to 30 rows.")
        balanced_rows.append(group.iloc[:30])  # at most 30 rows
        continue

    segments = find_one_segments(group['hand_in_pocket'].tolist())

    for (start, end) in segments:
        count = end - start + 1
        half = count // 2

        # Select window: half before + segment + half after
        win_start = max(0, start - half)
        win_end = min(len(group)-1, end + half)

        window = group.iloc[win_start:win_end+1]
        balanced_rows.append(window)

if balanced_rows:
    balanced_df = pd.concat(balanced_rows, ignore_index=True)
    balanced_df = balanced_df.drop(columns=['camera_num','video_num'])
    balanced_df.to_csv(output_csv, index=False)
    print(f"Balanced CSV saved to {output_csv}")
else:
    print("No rows matched the filter or no balancing required.")


# balanced_rows = []  # to collect final rows

# # Group by camera, video, desk
# for (cam, vid, desk), group in df.groupby(['camera_num', 'video_num', 'desk'], sort=False):
#     # Filter specific videos if provided
#     if videos_to_process is not None and vid not in videos_to_process:
#         continue

#     group = group.reset_index(drop=True)
#     ones_idx = group.index[group['hand_in_pocket'] == 1].tolist()

#     if len(ones_idx) == 0:
#         # No positives → check row count limit
#         if len(group) <= 30:
#             balanced_rows.append(group)
#         else:
#             print(f"WARNING: {cam=}, {vid=}, {desk=} has {len(group)} rows and no positives → skipped.")
#         continue

#     # Identify continuous segments of '1's
#     segments = []
#     start = ones_idx[0]
#     prev = start
#     for idx in ones_idx[1:]:
#         if idx == prev + 1:
#             prev = idx
#         else:
#             segments.append((start, prev))
#             start = idx
#             prev = idx
#     segments.append((start, prev))  # last segment

#     # Collect rows around each segment
#     selected_indices = set()
#     for seg_start, seg_end in segments:
#         seg_len = seg_end - seg_start + 1
#         half = seg_len // 2

#         # Middle point of the segment
#         mid = (seg_start + seg_end) // 2

#         # Select window around mid: half rows up and down
#         start_idx = max(0, mid - half)
#         end_idx = min(len(group) - 1, mid + half)

#         for i in range(start_idx, end_idx + 1):
#             selected_indices.add(i)

#     # Collect balanced rows for this group
#     balanced_rows.append(group.loc[sorted(selected_indices)])

# # --- Combine and save ---
# if balanced_rows:
#     final_df = pd.concat(balanced_rows).sort_values(by=['camera_num','video_num','desk','frame'])
#     # Drop helper columns before saving
#     final_df = final_df.drop(columns=['camera_num', 'video_num'])
#     final_df.to_csv(output_csv, index=False)
#     print(f"Balanced CSV saved to {output_csv}")
# else:
#     print("No rows selected → output CSV not created.")
