import pandas as pd
import os

input_csv = "temp_keypoint_l3"

df = pd.read_csv(f"C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/{input_csv}.csv")

x_cols = [ col for col in df.columns if "_x" in col ]
y_cols = [ col for col in df.columns if "_y" in col ]

df[x_cols] = df[x_cols].applymap(lambda x: x/1280 if x != 0 else -1)
df[y_cols] = df[y_cols].applymap(lambda y: y/720 if y != 0 else -1)

df.to_csv(f"C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/{input_csv}_norm.csv", index=False)

print(f"Normalized keypoints saved to {input_csv}_norm.csv")