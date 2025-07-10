import cv2
from ultralytics import YOLO
import requests
import json
import numpy as np
import os
from datetime import datetime
import torch

# Initialize YOLO-World and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("yolov8s-world.pt").to(device)

# Define a comprehensive list of common objects for real-time detection
common_classes = [
    "cup", "bottle", "chair", "table", "book", "phone", "laptop", "pen", "paper", 
    "keyboard", "mouse", "monitor", "desk", "lamp", "plate", "fork", "spoon", 
    "knife", "bowl", "mug", "glass", "teapot", "clock", "vase", "plant", "shoe", 
    "bag", "hat", "shirt", "pants", "jacket", "umbrella", "backpack", "wallet"
]  # Add more classes as needed
model.set_classes(common_classes)  # Set model to detect all common classes

# xAI API setup
API_KEY = os.getenv("XAI_API_KEY", "xai-1UYtI5dpr0ql2K8xkdy8rrYnxUYTyXDVor6yGXYNu00H1fMZtTuKJFk6lmbV5t0ub9VdGaekHg6JZucB")
API_URL = "https://api.x.ai/v1/chat/completions"

def query_grok3(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "grok-3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    frame_resized = cv2.resize(frame, (640, 640))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Run YOLO-World inference
    results = model.predict(frame_rgb, conf=0.5, device=device)
    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf
    labels = results[0].boxes.cls

    # Extract detections
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        class_name = common_classes[int(label)]  # Map label index to class name
        x1, y1, x2, y2 = map(int, box)
        detections.append(f"{class_name} detected at coordinates ({x1},{y1},{x2},{y2}) with confidence {score:.2f}")
    detection_text = "; ".join(detections) if detections else "No objects detected."

    # Query Grok 3
    prompt = f"Describe the scene where: {detection_text}. Provide a human-like description of the objects detected."
    grok_response = query_grok3(prompt)

    # Visualize detections
    for box, score, label in zip(boxes, scores, labels):
        class_name = common_classes[int(label)]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_resized, f"{class_name}: {score:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display Grok response
    cv2.putText(frame_resized, grok_response[:50] + "...", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow("Real-Time Detection", frame_resized)
    print("Grok 3 Response:", grok_response)

    # Capture image on 'Enter' key press (key code 13)
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter key
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        cv2.imwrite(filename, frame_resized)
        print(f"Image saved as {filename}")

        # Pause and prompt for object to detect
        cap.release()
        cv2.destroyAllWindows()
        object_to_detect = input("Enter the object you want to detect (e.g., cup, bottle): ").strip()

        # Set model to detect only the user-specified object
        try:
            model.set_classes([object_to_detect])
        except RuntimeError as e:
            print(f"Error setting classes: {e}")
            break

        # Load and process the captured image
        captured_image = cv2.imread(filename)
        if captured_image is None:
            print(f"Error: Could not load image {filename}.")
            break
        captured_resized = cv2.resize(captured_image, (640, 640))
        captured_rgb = cv2.cvtColor(captured_resized, cv2.COLOR_BGR2RGB)

        # Run YOLO-World inference on captured image
        results = model.predict(captured_rgb, conf=0.5, device=device)
        boxes = results[0].boxes.xyxy
        scores = results[0].boxes.conf
        labels = results[0].boxes.cls

        # Extract detections for the specified object only
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            detections.append(f"{object_to_detect} detected at coordinates ({x1},{y1},{x2},{y2}) with confidence {score:.2f}")
        detection_text = "; ".join(detections) if detections else f"No {object_to_detect}s detected."

        # Query Grok 3 for the captured image
        prompt = f"Describe the scene where: {detection_text}. Provide a human-like description or answer questions about the {object_to_detect}."
        grok_response = query_grok3(prompt)

        # Visualize only the specified object detections on captured image
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(captured_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(captured_resized, f"{object_to_detect}: {score:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the final image with only the specified object highlighted
        final_filename = f"detected_{object_to_detect}_{timestamp}.jpg"
        cv2.imwrite(final_filename, captured_resized)
        print(f"Final image saved as {final_filename}")

        # Display the final image
        cv2.imshow(f"{object_to_detect} Detection on Captured Image", captured_resized)
        print("Grok 3 Response for Captured Image:", grok_response)

        # Wait for user to close the image window and exit
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break

    if key == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
