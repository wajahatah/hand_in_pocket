""" loads the keypoint from the csv and crops the images from the given root directory
    - append the crops and keypoints as a temporal sequence of 5 frames"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class TemporalKeypointCropDataset(Dataset):
    def __init__(self, csv_file, crop_root, window_size=5):
        self.df = pd.read_csv(csv_file)
        self.crop_root = crop_root
        self.window_size = window_size
        self.samples = []

        for i in range(len(self.df) - window_size + 1):
            window = self.df.iloc[i:i + window_size]
            if (
                len(window['source_file'].unique()) == 1 and
                len(window['desk_no'].unique()) == 1 and
                np.all(np.diff(window['frame'].values) == 1) and
                window[['kp_0_x','kp_0_y','kp_1_x','kp_1_y']].notnull().all().all()
            ):
                # self.samples.append(window)
                source_file = window.iloc[0]['source_file'].replace('.csv', '')
                desk_no = int(window.iloc[0]['desk_no'])
                all_images_exist = True
                for _, row in window.iterrows():
                    frame = int(row['frame'])
                    img_name = f"{source_file}_d{desk_no}_{frame}.jpg"
                    if not any(os.path.exists(os.path.join(self.crop_root, folder, img_name)) for folder in ["hand_in_pocket", "no_hand_in_pocket"]):
                        all_images_exist = False
                        print(f"Missing crop for {img_name}")
                        # break
                        continue
                if all_images_exist:
                    self.samples.append(window)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window = self.samples[idx]
        source_file = window.iloc[0]['source_file'].replace('.csv', '')
        desk_no = int(window.iloc[0]['desk_no'])
        position = window.iloc[0]['position']

        crops = []
        keypoints = []

        for _, row in window.iterrows():
            frame = int(row['frame'])
            img_name = f"{source_file}_d{desk_no}_{frame}.jpg"
            for folder in ["hand_in_pocket", "no_hand_in_pocket"]:
                img_path = os.path.join(self.crop_root, folder, img_name)
                if os.path.exists(img_path):
                    break
            else:
                raise FileNotFoundError(f"Crop not found: {img_name}")

            img = Image.open(img_path).convert('L').resize((64, 64))
            crops.append(np.array(img, dtype=np.float32) / 255.0)
            keypoints.append(row[4:14].tolist() + row[14:24].tolist())  # 10 x, 10 y

        crops = torch.tensor(np.stack(crops)).unsqueeze(1)  # (5, 1, 64, 64)
        keypoints = torch.tensor(np.stack(keypoints), dtype=torch.float32).view(-1)  # (100,)
        keypoints = torch.cat([keypoints, torch.tensor([position], dtype=torch.float32)])  # (101,)

        label = int(window.iloc[self.window_size // 2]['hand_in_pocket'])
        return crops, keypoints, torch.tensor(label, dtype=torch.long)