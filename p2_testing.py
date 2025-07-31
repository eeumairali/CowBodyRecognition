from ultralytics import YOLO
import os

# === Paths ===
ubuntu_path = "/mnt/c/Users/eeuma/Desktop/Cow_Body_based_recognition"
weights_path = os.path.join(ubuntu_path, "Trained_detection_models", "cowbody_detector6", "weights", "best.pt")
image_path = os.path.join(ubuntu_path, "CowBodyDetectionData-2", "train", "images", "121_jpg.rf.5402f6a09691c3953c83699686ab3c12.jpg")
output_dir = os.path.join(ubuntu_path, "test_output", "predictions")

# === Load model ===
model = YOLO(weights_path)

# === Predict on single image ===
results = model.predict(
    source=image_path,
    save=True,
    save_txt=True,
    save_conf=True,
    project=os.path.dirname(output_dir),
    name=os.path.basename(output_dir)
)

print(f"✅ Output saved to: {output_dir}")
