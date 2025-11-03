import os
import shutil

# === CONFIG ===
root_dir = "C:/wajahat/hand_in_pocket/dataset/images_bb/batch1_split/train"  # parent directory containing all subfolders
output_images = "C:/wajahat/hand_in_pocket/dataset/images_bb/training1/train/images"
output_labels = "C:/wajahat/hand_in_pocket/dataset/images_bb/training1/train/labels"

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

counter = 0

# Loop through all subfolders
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    images_path = os.path.join(folder_path, "images")
    labels_path = os.path.join(folder_path, "new_labels")

    if not (os.path.isdir(images_path) and os.path.isdir(labels_path)):
        print(f"Skipping {folder_path}, missing images or labels folder.")
        continue

    # List all images
    for img_name in sorted(os.listdir(images_path)):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        
        # Base name without extension (e.g., "c1_v1_00")
        base_name = os.path.splitext(img_name)[0]

        # Find label file that ends with this base name
        matched_label = None
        for lbl_name in os.listdir(labels_path):
            if lbl_name.endswith(base_name + ".txt"):
                matched_label = lbl_name
                break

        if not matched_label:
            print(f"No matching label found for {img_name}")
            continue

        # Define new names
        new_img_name = f"f_{counter:04d}.jpg"
        new_lbl_name = f"f_{counter:04d}.txt"

        # Copy to output folder
        shutil.copy2(os.path.join(images_path, img_name),
                     os.path.join(output_images, new_img_name))
        shutil.copy2(os.path.join(labels_path, matched_label),
                     os.path.join(output_labels, new_lbl_name))

        counter += 1

print(f"✅ Done. Total pairs processed: {counter}")
print(f"Images saved to: {output_images}")
print(f"Labels saved to: {output_labels}")
