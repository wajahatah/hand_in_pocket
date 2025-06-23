""" Get frames from the folder and run keyoint model on them ans save the frames with keypoints drawn on them """
from ultralytics import YOLO
import os
import cv2

frame_folder = "F:/Wajahat/hand_in_pocket/frames/no_hp"
output_folder = "F:/Wajahat/hand_in_pocket/frames/kp_no_hp"
os.makedirs(output_folder, exist_ok=True)

img = [f for f in os.listdir(frame_folder) if f.endswith('.jpg') or f.endswith('.png')]

model = YOLO("C:/wajahat/hand_in_pocket/bestv7-2.pt")

for image in img:
    image_path = os.path.join(frame_folder, image)
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error reading {image_path}")
        continue

    results = model(frame)[0]
    if results.keypoints is None:
        print(f"No keypoints found in {image}")
        continue

    keypoints_tensor = results.keypoints.data
    for person_idx, kp_tensor in enumerate(keypoints_tensor):
        for i, keypoint in enumerate(kp_tensor[:10]):
            x, y, conf = keypoint[:3].cpu().numpy()
            if conf > 0.5:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imshow("Keypoints", frame)
    key = cv2.waitKey(1)

    output_path = os.path.join(output_folder, image)
    cv2.imwrite(output_path, frame)
    print(f"Processed {image}, saved to {output_path}")
