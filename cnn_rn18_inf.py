import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ========== CONFIG ==========
video_path = "C:/wajahat/hand_in_pocket/test_bench/cam_1_t1.mp4"
# output_path = 'output_crossentropy.mp4'
model_path = "C:/wajahat/hand_in_pocket/rf_models/cnn_rn_sig_aug.pth"
img_size = 640
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# ========== LOAD MODEL (TWO OUTPUT NEURONS) ==========
# model = models.resnet18(pretrained=False)
# model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.to(device)
# model.eval()

# def predict_frame(frame):
#     img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     img_tensor = transform(img_pil).unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = model(img_tensor)  # shape: [1, 2]
#         probs = torch.softmax(output, dim=1).squeeze(0)  # shape: [2]
#         prediction = torch.argmax(probs).item()
#         confidence = probs[prediction].item()
#         label = "HAND IN POCKET" if prediction == 1 else "NO HAND IN POCKET"
#     return label, confidence

# ========== LOAD MODEL (One OUTPUT NEURON) ==========
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)  # 1 class
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def predict_frame(frame):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor).squeeze(1)
        prob = torch.sigmoid(output)
        confd = prob.item()  # shape: [1]
        prediction = 1 if prob > 0.5 else 0  
        label = "HAND IN POCKET" if prediction == 1 else "NO HAND IN POCKET"
    return label, confd 

# ========== LOAD VIDEO ==========
cap = cv2.VideoCapture(video_path)
# width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps    = cap.get(cv2.CAP_PROP_FPS)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# frame_count = 0
# print(f"🎥 Processing: {video_path}")
if not cap.isOpened():
    print("Error loading video.")
    exit()

# while cap.isOpened():
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))  # Resize to match model input size

    # Predict
    label, conf = predict_frame(frame)

    # Draw on frame
    text = f"{label} ({conf:.2f})"
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 2)

    # Write frame
    # out.write(frame)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # frame_count += 1
    # if frame_count % 10 == 0:
    #     print(f"Processed {frame_count} frames...", end='\r')

cap.release()
cv2.destroyAllWindows()
# out.release()
# print(f"\n✅ Saved output video to: {output_path}")
