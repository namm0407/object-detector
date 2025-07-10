# object-detector

## 🎯 Overview
The object detector is built using BLIP-2 and YOLOv8l-world. This project was inspired by Agentic Object Detection developed by Andrew Ng, a pioneer in the field of artificial intelligence and the founder of LandingAI. I have taking this idea and create my own from scratch. 

Agentic Object Detection is an AI-driven technology that identifies and locates specific objects in images or videos using text prompts, without requiring extensive model training. This is a significant advancement from traditional object detection methods, as it leverages reasoning capabilities to provide more accurate and efficient detection. 

### 💫 Key Features
* vision-language tasks: Uses BLIP-2 is an advanced AI model developed by Salesforce AI Research
* Real-time Object Detection: Uses YOLOv8l-world models for fast and accurate object detection
* full stack project: provided with a web page to handle real-time inventory updates, customer interactions, and other dynamic data updates, ensuring a seamless and responsive user experience.
* Backend: made with Python
* Frontend: made with React. The frontend handles the user interface, client-side logic, and data processing

## Demo video



## Limitations
This is a beginner-level project and currently exhibits lower accuracy due to its preliminary development stage. As an initial effort, the model relies on pretrained weights without extensive fine-tuning or optimization for the specific dataset, which limits its performance. Factors such as insufficiently diverse or poorly labeled training data, challenges in integrating BLIP-2’s vision-language outputs with YOLOv8’s detection pipeline, and difficulties in detecting small objects contribute to reduced accuracy.


## How to run this
### 1. Clone the repository

`git clone https://github.com/namm0407/object-detector-with-chatbox.git`

`cd object-detector-with-chatbox`

### 2. Create virtual environment
Open terminal and type

`python -m venv env`

`env\Scripts\activate # on Mac source env/bin/activate`

### 3. Install Dependencies (if you haven't already)

#### type these in terminal

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

`pip install ultralytics`

`pip install Pillow`

`pip install flask`

`pip install flask-cors`

### 4. Install the models (if you haven't already)

#### For YOLO-World Model (yolov8l-world)

Run this in a python file

```bash

from ultralytics import YOLO

model = YOLO('yolov8l-world.pt')
```

#### For BLIP-2 Model
```bash
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load processor and model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")
```

### 5. create babel.min.js in the frontend folder 📂  
Open your browser and navigate to: https://cdn.jsdelivr.net/npm/@babel/standalone@7.27.7/babel.min.js

Right-click the page and select Save As
Save the file as babel.min.js in

### 6. run the code
#### To run the backend
`python main.py`

#### To run the frontend
`cd "C:\your\path\object detector\frontend"`

e.g.
`cd "C:\Users\Public\Documents\object detector\frontend"`

`code .`
`python -m http.server 3000`
