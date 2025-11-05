"""delete specific classes from the label file to train the yolo model"""

import os
from pathlib import Path
import shutil

# === USER CONFIG ===
labels_src = Path("C:/wajahat/hand_in_pocket/dataset/images_bb/training1/labels")     # original labels folder
labels_dst = Path("C:/wajahat/hand_in_pocket/dataset/images_bb/training1/revised_labels")      # new folder for cleaned labels
remove_classes = {4, 7, 9}                       # class IDs you want to delete
# ====================

# Recreate destination folder
if labels_dst.exists():
    shutil.rmtree(labels_dst)
labels_dst.mkdir(parents=True, exist_ok=True)

# Walk through label files
for txt_file in labels_src.rglob("*.txt"):
    rel_path = txt_file.relative_to(labels_src)
    dst_file = labels_dst / rel_path
    dst_file.parent.mkdir(parents=True, exist_ok=True)

    lines = txt_file.read_text().strip().splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        cls_id = int(parts[0])
        if cls_id in remove_classes:
            continue  # skip unwanted classes
        new_lines.append(line)

    # Save updated file
    if new_lines:
        dst_file.write_text("\n".join(new_lines))
    else:
        dst_file.write_text("")  # write empty file if all lines removed

print(f"✅ Done. Removed classes {remove_classes}. Cleaned labels saved to: {labels_dst}")
