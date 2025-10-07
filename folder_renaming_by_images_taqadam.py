import os
import re

# === CONFIGURATION ===
root_dir = "C:/wajahat/hand_in_pocket/dataset/images_bb"  # Change this to your root folder

# === FUNCTION TO EXTRACT PREFIX (before last underscore) ===
def get_prefix(filename):
    match = re.match(r"^(.*)_[0-9]+", os.path.splitext(filename)[0])
    return match.group(1) if match else None

# === MAIN LOOP ===
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Collect all valid image filenames
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        print(f"⚠️ No images found in {folder_name}")
        continue

    # Extract prefixes
    prefixes = {get_prefix(img) for img in images if get_prefix(img)}

    if len(prefixes) == 1:
        prefix = list(prefixes)[0]
        new_folder_name = os.path.basename(prefix)  # e.g. c1_v3
        new_folder_path = os.path.join(root_dir, new_folder_name)

        # Avoid overwriting existing folders
        if new_folder_path == folder_path:
            print(f"✅ Already named correctly: {folder_name}")
        elif os.path.exists(new_folder_path):
            print(f"⚠️ Skipped: {new_folder_name} already exists.")
        else:
            os.rename(folder_path, new_folder_path)
            print(f"✅ Renamed '{folder_name}' → '{new_folder_name}'")
    else:
        print(f"❌ Inconsistent prefixes in {folder_name}: {prefixes}")

print("\n🎯 Folder renaming complete!")
