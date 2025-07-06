from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLOWorld
from PIL import Image, ImageDraw
from collections import defaultdict
import os
import torch
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize YOLO-World
yolo_model = YOLOWorld('yolov8m-world.pt')

# Ensure upload and output directories exist
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.png')
    file.save(image_path)

    # Process image with YOLO
    image = Image.open(image_path).convert("RGB")
    results = yolo_model.predict(image)
    detections = results[0].boxes.data.tolist()

    # Count detected objects for caption
    object_counts = defaultdict(int)
    for det in detections:
        _, _, _, _, conf, cls_id = det
        if conf > 0.5:  # Filter low-confidence detections
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
    image.save(output_path)

    return jsonify({
        'caption': caption,
        'image_url': '/api/image/annotated_image.jpg',
        'detections': [{'class': yolo_model.names[int(cls_id)], 'conf': conf, 'bbox': [x1, y1, x2, y2]} for x1, y1, x2, y2, conf, cls_id in detections if conf > 0.5]
    })

@app.route('/api/search', methods=['POST'])
def search_object():
    data = request.get_json()
    target_object = data.get('target_object', '').lower()
    if not target_object:
        return jsonify({'error': 'No object specified'}), 400

    # Reload original image
    image_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.png')
    if not os.path.exists(image_path):
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

    # Save image with target object
    output_filename = f"{target_object}.jpg"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    image.save(output_path)

    return jsonify({
        'found': found,
        'image_url': f'/api/image/{output_filename}' if found else None,
        'message': f"{target_object} {'found' if found else 'not found'} in the image"
    })

@app.route('/api/image/<filename>', methods=['GET'])
def serve_image(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
