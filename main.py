from ultralytics import YOLO

# Load Model
model = YOLO("yolov8n.yaml") #build a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=1) # train the model
