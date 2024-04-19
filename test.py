import torch
from PIL import Image
from ultralytics import YOLO

# Load the trained model
model_path = 'yolov8n.pt'
model = YOLO(model_path)

# Load an image
image_path = 'agri_data/data/Validation_Data/agri_0_2077.jpeg'
image = Image.open(image_path)

# Perform inference
results = model(image)

# Check and handle the type of results
if isinstance(results, list):
    for result in results:
        print(result)  # Assuming each result item can be printed directly
else:
    results.print()  # Use this if the results are not a list and can be handled by built-in methods

# Define the output directory
output_dir = 'runs/detect/predict'

# Assuming results have a method to save
if hasattr(results, 'save'):
    results.save(save_dir=output_dir)
