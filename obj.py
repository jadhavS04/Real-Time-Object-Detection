from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("yolo11x.pt")  # Use appropriate model name

# Run prediction using webcam
results = model.predict(source=0, show=True)

# Print results (optional, prints once)
print(results)
