# object-detector

# object-detector-with-chatbox

* made with BLIP-2 & yolov8l-world

* need to press ctri + shift + p to switch to blip env

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
