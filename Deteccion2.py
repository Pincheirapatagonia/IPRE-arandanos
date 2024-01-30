
import os
os.environ['NUMEXPR_MAX_THREADS'] = '12'  # Or any other number you want

import cv2
import numpy as np
import threading
import torch
from torchvision import transforms
from ultralytics import YOLO


fourcc = cv2.VideoWriter_fourcc(*'XVID')
# Adjust fps and frame size
out = cv2.VideoWriter('output.avi', fourcc, 1.0, (640, 480))

class Tracker:
    def __init__(self, model):
        
        self.record = True
        self.cap = None
        self.image = None
        self.out = cv2.VideoWriter('output.avi', fourcc, 1.0, (640, 480))  # Move this line inside the Tracker class
        self.tracking = True
        self.show = True
        self.model = model
        self.detect = False
        self.camera_num = 0
        self.video_file = "input.mp4"
        
    def initiateVideo(self):
        self.cap = cv2.VideoCapture(self.video_file)  # Modify this line
        if not self.cap.isOpened():  # Check if the video file was successfully opened
            print("Error opening video file")
            return
        ret, frame = self.cap.read()
        while (not ret):
            print(ret, frame)
            ret, frame = self.cap.read()
        self.y_max, self.x_max, _ = frame.shape
        self.obj = [int(self.x_max / 2), int(self.y_max)]

    def track(self):
        while self.tracking:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:  # Check if the frame was successfully read
                self.tracking = False
                break
            # Make predictions
            with torch.no_grad():
                predictions = self.model(frame)
                a = False
                for result in predictions:
                    print(result)
                    result_bboxes = result.boxes
                    for result_bbox in result_bboxes:
                        b_coordinates = result_bbox.xyxy[0]
                        # get box coordinates in (top, left, bottom, right) format
                        b_center = result_bbox.xywh[0]
                        box = b_coordinates[:4].int().tolist()
                        confidence = result_bbox.conf
                        if confidence[0].cpu().numpy() > 0:
                            a = True
                            if (self.record or self.show):
                                frame = cv2.circle(frame, (int(b_center[0].cpu().numpy()), int(
                                    b_center[1].cpu().numpy())), 5, (0, 0, 255), -1)
                                frame = cv2.rectangle(
                                    frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                                label = "Racimo_Maduro"
                                frame = cv2.putText(
                                    frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            break

                if (self.record):
                    frame = cv2.line(frame, (int(
                        self.x_max / 2) - 70, int(self.y_max)), (int(self.x_max / 2) - 70, 0), (255, 0, 0), 2)
                    frame = cv2.line(frame, (int(
                        self.x_max / 2) + 70, int(self.y_max)), (int(self.x_max / 2) + 70, 0), (255, 0, 0), 2)
                    self.out.write(frame)  # Use self.out instead of out
                self.detect = a
            if self.show:
                frame = cv2.line(frame, (int(self.x_max / 2) - 70, int(self.y_max)),
                                (int(self.x_max / 2) - 70, 0), (255, 0, 0), 2)
                frame = cv2.line(frame, (int(self.x_max / 2) + 70, int(self.y_max)),
                                (int(self.x_max / 2) + 70, 0), (255, 0, 0), 2)
                cv2.imshow('Object Detection', frame)
                cv2.waitKey(1)

    def stop_tracking(self):
        self.tracking = False

    def finish(self):
        self.cap.release()
        self.out.release()  # Use self.out instead of out
        cv2.destroyAllWindows()


class Brain:

    def __init__(self, tracker) -> None:
        self.tracker = tracker
        self.tracking_thread = threading.Thread(
            target=self.track_wrapper, args=())
        self.tracking_thread.daemon = True
        self.begin()
        self.do()
    def track_wrapper(self):
        # This function is used to run the track() method in a separate thread.
        self.tracker.track()

    def begin(self):
        try:  # Add this line
            print("Starting...\n")
            self.tracker.initiateVideo()
            print("Video Started...\n")
            self.tracking_thread.start()
            print("Track thread Started...\n")
            print("Ready\n")
            while self.tracker.detect:  # Use self.tracker.detect in a condition
                pass  # Do something while self.tracker.detect is True
        finally:  # Add this line
            self.finish()
        
    def finish(self):
        # Close the serial port
        # Release the video writer after the main loop
        self.tracker.stop_tracking()
        self.tracker.finish()
        self.tracking_thread.join()
        
    def do(self):
        return self.tracker.track()

print("loading model...")

model = YOLO("ArandanosV1.pt")
print("model loaded")
brain = Brain(Tracker(model))
