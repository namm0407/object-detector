from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLOWorld
from PIL import Image, ImageDraw
from collections import defaultdict
from pathlib import Path
import os
import torch
import io
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize YOLO-World
yolo_model = YOLOWorld('yolov8m-world.pt')

# Define OUTPUT_FOLDER as an absolute path
BASE_DIR = Path(__file__).resolve().parent  # Directory of main.py
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Debug the absolute path
print(f"UPLOAD_FOLDER absolute path: {os.path.abspath(UPLOAD_FOLDER)}")
print(f"OUTPUT_FOLDER absolute path: {os.path.abspath(OUTPUT_FOLDER)}")

@app.route('/api/upload', methods=['POST'])
def upload_image():
    print("Received upload request")  # Debug start of upload
    if 'image' not in request.files:
        print("Error: No image provided in request")
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        print("Error: No file selected")
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.png')
    print(f"Attempting to save image at: {os.path.abspath(image_path)}")  # Debug save path
    try:
        file.save(image_path)
        print(f"Image saved successfully at: {os.path.abspath(image_path)}")
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return jsonify({'error': f'Failed to save image: {str(e)}'}), 500

    if not os.path.exists(image_path):
        print(f"Error: Image not found after saving at {os.path.abspath(image_path)}")
        return jsonify({'error': 'Failed to save uploaded image'}), 500

    # Process image with YOLO
    image = Image.open(image_path).convert("RGB")
    results = yolo_model.predict(image)
    detections = results[0].boxes.data.tolist()

    # Count detected objects for caption
    object_counts = defaultdict(int)
    for det in detections:
        _, _, _, _, conf, cls_id = det
        if conf > 0.5:
            object_counts[yolo_model.names[int(cls_id)]] += 1

    # Generate caption
    caption = "Detected: " + ", ".join([f"{count} {obj}" for obj, count in object_counts.items()]) if object_counts else "No objects detected"

    # Save annotated image with all detections
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        if conf > 0.5:
            label = f"{yolo_model.names[int(cls_id)]} {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), label, fill="red")
    output_path = os.path.join(OUTPUT_FOLDER, 'annotated_image.jpg')
    try:
        image.save(output_path)
        print(f"Annotated image saved at: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error saving annotated image: {str(e)}")
        return jsonify({'error': f'Failed to save annotated image: {str(e)}'}), 500

    return jsonify({
        'caption': caption,
        'image_url': '/api/image/annotated_image.jpg',
        'detections': [{'class': yolo_model.names[int(cls_id)], 'conf': conf, 'bbox': [x1, y1, x2, y2]} for x1, y1, x2, y2, conf, cls_id in detections if conf > 0.5]
    })

@app.route('/api/search', methods=['POST'])
def search_object():
    data = request.get_json()
    print(f"Received data: {data}")  # Debug incoming JSON
    target_object = data.get('target_object', '').lower()
    if not target_object:
        print("Error: No target_object specified")  # Debug specific error
        return jsonify({'error': 'No object specified'}), 400
    
    # Sanitize filename
    safe_filename = re.sub(r'[^a-zA-Z0-9]', '_', target_object)
    output_filename = f"{safe_filename}.jpg"
    
    # Reload original image
    image_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.png')
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")  # Debug image path
        return jsonify({'error': 'No image uploaded'}), 400

    image = Image.open(image_path).convert("RGB")
    results = yolo_model.predict(image)
    detections = results[0].boxes.data.tolist()

    # Draw boxes for target object
    draw = ImageDraw.Draw(image)
    found = False
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        if conf > 0.5 and yolo_model.names[int(cls_id)].lower() == target_object:
            label = f"{yolo_model.names[int(cls_id)]} {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), label, fill="red")
            found = True

    # Save image with sanitized filename
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    image.save(output_path)
    print(f"Image saved at: {os.path.abspath(output_path)}")
    if not os.path.exists(output_path):
        print(f"Error: File {os.path.abspath(output_path)} was not created")
        return jsonify({'error': 'Failed to save image'}), 500

    return jsonify({
        'found': found,
        'image_url': f'/api/image/{output_filename}' if found else None,
        'message': f"{target_object} {'found' if found else 'not found'} in the image"
    })

@app.route('/api/image/<filename>', methods=['GET'])
def serve_image(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    absolute_file_path = os.path.abspath(file_path)
    print(f"Attempting to serve file: {absolute_file_path}")
    if not os.path.exists(absolute_file_path):
        print(f"File not found: {absolute_file_path}")
        return jsonify({'error': f'Image {filename} not found'}), 404
    return send_file(absolute_file_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
