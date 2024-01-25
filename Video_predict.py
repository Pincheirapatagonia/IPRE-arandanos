import cv2
import torch
from torchvision.transforms import functional as F

# Load the YOLOv8 model
model = torch.hub.load_state_dict_from_url('file://ArandanosV1.pt', map_location=torch.device('cpu'))

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
    tensor = F.to_tensor(frame).unsqueeze(0)

    # Pass the tensor through the YOLOv8 model
    preds = model(tensor)

    # For each prediction, draw a bounding box on the frame
    for pred in preds.xyxy[0]:
        x1, y1, x2, y2 = map(int, pred[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

# Release the VideoCapture and VideoWriter
cap.release()
out.release()