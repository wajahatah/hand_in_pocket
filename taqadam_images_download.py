# import os
# import json
# import requests
# from tqdm import tqdm

# # === CONFIGURATION ===
# json_path = "C:/Users/LT/Downloads/Annotated images project 3964.json"
# output_dir = "C:/wajahat/hand_in_pocket/dataset/images_bb/t1"  # Folder where images will be saved
# timeout_sec = 15                # Max wait time for each image

# # === PREPARE OUTPUT DIRECTORY ===
# os.makedirs(output_dir, exist_ok=True)

# # === LOAD JSON FILE ===
# with open(json_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Handle both list and dict formats
# if isinstance(data, dict):
#     data = [data]

# print(f"🧾 Found {len(data)} image entries in JSON")

# # === DOWNLOAD LOOP ===
# for i, item in enumerate(tqdm(data, desc="Downloading images")):
#     # Try to get image URL
#     image_url = item.get("image_url") or item.get("url") or item.get("image")
#     if not image_url:
#         print(f"⚠️  No image_url found for entry {i}, skipping.")
#         continue

#     # Try to get image name (fall back to last part of URL)
#     image_name = item.get("image_name") or os.path.basename(image_url.split("?")[0])
#     save_path = os.path.join(output_dir, image_name)

#     # Skip already-downloaded images
#     if os.path.exists(save_path):
#         continue

#     try:
#         response = requests.get(image_url, stream=True, timeout=timeout_sec)
#         if response.status_code == 200:
#             with open(save_path, "wb") as f:
#                 for chunk in response.iter_content(8192):
#                     f.write(chunk)
#         else:
#             print(f"⚠️ Failed to download {image_name}: HTTP {response.status_code}")
#     except Exception as e:
#         print(f"❌ Error downloading {image_name}: {e}")

# print(f"\n✅ All available images downloaded to '{output_dir}'")


import os
import requests
from tqdm import tqdm

# === CONFIGURATION ===
txt_path = "C:/Users/LT/Downloads/original_images_url_project_3964 (1).txt"     # Path to your text file
output_dir = "C:/wajahat/hand_in_pocket/dataset/images_bb/t1"     # Where to save images
timeout_sec = 20                     # Timeout for each request (seconds)

# === CREATE OUTPUT FOLDER ===
os.makedirs(output_dir, exist_ok=True)

# === READ ALL URLS ===
with open(txt_path, "r", encoding="utf-8") as f:
    urls = [line.strip().strip('"') for line in f if line.strip()]

print(f"🧾 Found {len(urls)} image URLs in {txt_path}")

# === DOWNLOAD LOOP ===
for url in tqdm(urls, desc="Downloading images"):
    # Extract filename (remove query parameters)
    filename = os.path.basename(url.split("?")[0])
    save_path = os.path.join(output_dir, filename)

    # Skip if already downloaded
    if os.path.exists(save_path):
        continue

    try:
        response = requests.get(url, stream=True, timeout=timeout_sec)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
        else:
            print(f"⚠️ Failed {filename}: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")

print("\n✅ Download complete! Check your 'downloaded_images' folder.")
