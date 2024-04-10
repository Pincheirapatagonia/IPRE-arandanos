import cv2
import torch
from torchvision.transforms import functional as F
from ultralytics import YOLO
import os

# Check if CUDA is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
cv2.setNumThreads(0)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
# Load the YOLOv8 model
# Make sure to provide the correct path to your model file
model_path = 'ArandanosV1.pt'  # Update this to your actual model path
#model = torch.load(model_path, map_location=device)
model = YOLO(model_path)
#model = model['model'].float()
# Move the model to the device
#model = model.to(device)

# Open the input video file
cap = cv2.VideoCapture('input.mp4')

# Get the video's frame rate and size
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter for the output video
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PyTorch tensor
    #tensor = F.to_tensor(frame).unsqueeze(0)

    # Move the tensor to the device
    #tensor = tensor.to(device)

    # Pass the tensor through the YOLOv8 model
    #preds = model(tensor)
    preds = model(frame)
    # For each prediction, draw a bounding box on the frame
    for pred in preds.xyxy[0]:
        x1, y1, x2, y2 = map(int, pred[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

# Release the VideoCapture and VideoWriter
cap.release()
out.release()