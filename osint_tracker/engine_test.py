from ultralytics import YOLO

# 1. Initialize the YOLO architecture and request the Nano weights
print("Initializing architecture and seeking weights...")
model = YOLO("yolov8n.pt")

# 2. Output the mathematical structure of the model to prove it loaded
print("\n--- ENGINE DIAGNOSTIC ---")
print(f"Model Architecture: {model.task}")
print(f"Model Weights Loaded: {model.ckpt_path}")
print("Engine is hot and ready for inference.") 