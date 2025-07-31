import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image
import tempfile
import shutil
import cv2
import numpy as np

# === Setup paths ===
ubuntu_path = "/mnt/c/Users/eeuma/Desktop/Cow_Body_based_recognition"
weights_path = os.path.join(ubuntu_path, "Trained_detection_models", "cowbody_detector6", "weights", "best.pt")
output_dir = os.path.join(ubuntu_path, "streamlit_results")

# === Load model once ===
model = YOLO(weights_path)
st.title("🐄 Cow Body Detector - YOLOv8")

# === Upload image ===
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # === Save temp file ===
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_img_path = tmp_file.name

    st.image(temp_img_path, caption="📥 Uploaded Image", use_column_width=True)

    # === Run YOLOv8 prediction (no saving default image) ===
    results = model.predict(
        source=temp_img_path,
        save=False,
        save_txt=True,
        save_conf=True
    )

    # === Load original image using OpenCV for custom drawing ===
    image = cv2.imread(temp_img_path)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label with small font
            font_scale = 0.5
            thickness = 1
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(image, (x1, y1 - h - 4), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # === Save manually drawn result ===
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    result_path = os.path.join(output_dir, "predictions", f"result_{os.path.basename(temp_img_path)}")
    cv2.imwrite(result_path, image)

    # === Display result in Streamlit ===
    st.image(result_path, caption="✅ Prediction Result", use_column_width=True)

    # Optional: Clean up
    os.remove(temp_img_path)
