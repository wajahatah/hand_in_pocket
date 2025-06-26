import zipfile
import os

def unzip_file(zip_path, target_folder):
    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
        print(f"✅ Extracted '{zip_path}' to '{target_folder}'")

# ======== CONFIG ========
zip_path = '/home/ubuntu/wajahat/hp/hp.zip'  # e.g., 'C:/data/archive.zip'
target_folder = '/home/ubuntu/wajahat/hp/without_kp_frames'  # e.g., 'C:/data/unzipped'

# ======== RUN ========
unzip_file(zip_path, target_folder)
