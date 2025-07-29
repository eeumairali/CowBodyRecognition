from ultralytics import YOLO
import os
import pandas as pd

# === Define paths ===
ubuntu_path = "/mnt/c/Users/eeuma/Desktop/Cow_Body_based_recognition"
data_yaml = os.path.join(ubuntu_path, "CowBodyDetectionData-2/data.yaml")
project_dir = os.path.join(ubuntu_path, "Trained_detection_models")
run_name = "cowbody_detector"
metrics_csv_path = os.path.join(project_dir, run_name, "metrics.csv")
output_csv_path = os.path.join(ubuntu_path, "cow_training_summary.csv")

# === Load and train the model ===
model = YOLO("yolov8s.pt")
model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    name=run_name,
    project=project_dir,
    device=0
)

# === After training, extract training results from metrics.csv ===
if os.path.exists(metrics_csv_path):
    df = pd.read_csv(metrics_csv_path)
    columns = [
        "epoch", 
        "train/box_loss", "train/cls_loss", 
        "val/box_loss", "val/cls_loss", 
        "metrics/precision", "metrics/recall", 
        "metrics/mAP_0.5", "metrics/mAP_0.5:0.95"
    ]
    df_filtered = df[columns]
    df_filtered.to_csv(output_csv_path, index=False)
    print(f"✅ Training summary saved to {output_csv_path}")
else:
    print("⚠️ Training complete but metrics.csv not found.")
