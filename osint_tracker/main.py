import cv2
import os
from ultralytics import YOLO

# --- PHASE A & B: WAKE THE ENGINE ---
print("Waking up AI Engine...")
model = YOLO("yolov8n.pt")

# 1. Define the target directory (not a single file)
data_folder = "data"

# 2. Command the OS to list every file in that directory
try:
    image_files = os.listdir(data_folder)
    print(f"Target acquired: {len(image_files)} items found in '{data_folder}'.")
except FileNotFoundError:
    print(f"CRITICAL ERROR: Folder '{data_folder}' not found.")
    exit()

# --- PHASE C: THE AUTOMATION LOOP ---
for filename in image_files:
    # Defensive Logic 1: Ignore anything that isn't a standard image file
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Construct the exact path to the current image
    image_path = os.path.join(data_folder, filename)
    print(f"\nProcessing: {filename}")

    # Load the matrix
    image_matrix = cv2.imread(image_path)

    # Defensive Logic 2: If the image is corrupted, skip it and keep the pipeline alive
    if image_matrix is None:
        print(f"WARNING: Asset corrupted at {image_path}. Skipping.")
        continue

    # --- THE INFERENCE STRIKE ---
    results = model(image_matrix)
    detections = results[0].boxes

    # --- THE TRANSLATION & RENDER ---
    # --- THE TRANSLATION & RENDER ---
    for box in detections:
        # 1. Extract the Class ID and Confidence first
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # --- THE TACTICAL FILTER ---
        # ONLY proceed if the object is an airplane (4) AND the engine is >50% sure
        if class_id == 4 and confidence > 0.01:
            
            # If it passes the filter, extract coordinates and draw
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cv2.rectangle(image_matrix, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"Target Lock: {confidence:.2f}"
            cv2.putText(image_matrix, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # --- PIPELINE CONTROL ---
    cv2.imshow("OSINT Tactical View", image_matrix)

    # Wait for user input. 
    # 0xFF is a bitmask ensuring cross-platform keyboard compatibility.
    key = cv2.waitKey(0) & 0xFF

    # If the user presses 'q', break the loop and abort.
    if key == ord('q'):
        print("Manual abort initiated. Terminating pipeline.")
        break

# Clean up memory when the loop is done
cv2.destroyAllWindows()
print("\nPipeline execution completed.")