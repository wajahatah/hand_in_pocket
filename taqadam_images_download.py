import os
import requests
from tqdm import tqdm

# === CONFIGURATION ===
# txt_path = "C:/Users/LT/Downloads/original_images_url_project_3964 (1).txt"     # Path to your text file
txt_folder = "C:/Users/LT/Downloads/original_images_url_project_group_113 (1)/media/images"
output_dir = "C:/wajahat/hand_in_pocket/dataset/images_bb/batch3"     # Where to save images
timeout_sec = 20                     # Timeout for each request (seconds)

# === CREATE OUTPUT FOLDER ===
os.makedirs(output_dir, exist_ok=True)

txt_files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]

if not txt_files:
    print("❌ No .txt files found in the folder.")

for txt_file in txt_files:

    txt_path = os.path.join(txt_folder, txt_file)
    base_name = os.path.splitext(txt_file)[0]
    save_folder = os.path.join(output_dir, base_name)

    os.makedirs(save_folder, exist_ok=True)
# === READ ALL URLS ===
    with open(txt_path, "r", encoding="utf-8") as f:
        urls = [line.strip().strip('"') for line in f if line.strip()]

    print(f"🧾 Found {len(urls)} image URLs in {txt_path}")

    # === DOWNLOAD LOOP ===
    for url in tqdm(urls, desc="Downloading images"):
        # Extract filename (remove query parameters)
        filename = os.path.basename(url.split("?")[0])
        # save_path = os.path.join(output_dir, filename)
        save_path = os.path.join(save_folder, filename)

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

    print(f"✅ Finished '{txt_file}' — Saved in: {save_folder}\n")

print("\n🎉 All downloads complete!")

# print("\n✅ Download complete! Check your 'downloaded_images' folder.")
