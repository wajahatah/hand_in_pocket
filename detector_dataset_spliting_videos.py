import os
import random
import shutil

# === CONFIG ===
parent_dir = "C:/wajahat/hand_in_pocket/dataset/images_bb/batch1"  # path containing your ~50 folders
output_dir = "C:/wajahat/hand_in_pocket/dataset/images_bb/batch1_split"        # where new split folders will be created

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

# === CREATE OUTPUT STRUCTURE ===
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# === LIST AND SHUFFLE FOLDERS ===
all_folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
random.shuffle(all_folders)
total = len(all_folders)

# === COMPUTE SPLITS ===
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_folders = all_folders[:train_end]
val_folders = all_folders[train_end:val_end]
test_folders = all_folders[val_end:]

# === MOVE FOLDERS TO SPLIT DIRECTORIES ===
def move_folders(folders, dest_dir):
    for folder in folders:
        src = os.path.join(parent_dir, folder)
        dst = os.path.join(dest_dir, folder)
        print(f"Moving: {src} → {dst}")
        shutil.move(src, dst)

move_folders(train_folders, train_dir)
move_folders(val_folders, val_dir)
move_folders(test_folders, test_dir)

print(f"\n✅ Split completed successfully:")
print(f"  Train: {len(train_folders)} folders")
print(f"  Val:   {len(val_folders)} folders")
print(f"  Test:  {len(test_folders)} folders")
