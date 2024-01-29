import math
import cv2
import numpy as np
import serial
import time
# import keyboard
import threading
import torch
from torchvision import transforms
from ultralytics import YOLO
import csv

fourcc = cv2.VideoWriter_fourcc(*'XVID')
# Adjust fps and frame size
out = cv2.VideoWriter('output.avi', fourcc, 1.0, (640, 480))

class NutsTracker:
    def __init__(self, model):
        self.record = True
        self.cap = None
        self.image = None
        self.distancia = "0"
        self.area = "0"
        self.tracking = True
        self.show = True
        self.x = -1
        self.y = -1
        self.x_max = 0
        self.y_max = 0
        self.model = model
        self.detect = False
        self.obj = [0, 0]
        self.camera_num = 0

    def initiateVideo(self):
        self.cap = cv2.VideoCapture(self.camera_num)
        while not self.cap.isOpened():
            print("Error opening video")
            self.cap = cv2.VideoCapture(self.camera_num)
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
            # Make predictions
            with torch.no_grad():
                predictions = self.model(frame)
                a = False
                min_dist = 10000000000000000000000000
                best_x = 0
                best_y = 0
                for result in predictions:
                    result_bboxes = result.boxes
                    # print(result)
                    for result_bbox in result_bboxes:
                        b_coordinates = result_bbox.xyxy[0]
                        too_much_overlap = False
                        # get box coordinates in (top, left, bottom, right) format
                        b_center = result_bbox.xywh[0]
                        box = b_coordinates[:4].int().tolist()
                        # for other in result_bboxes:
                        #    o_coordinates = other.xyxy[0]
                        #    if (b_coordinates.cpu().numpy() == o_coordinates.cpu().numpy()).all():
                        #        continue
                        # Calculate the intersection coordinates
                        #    x1 = max(b_coordinates[0].cpu().numpy(), o_coordinates[0].cpu().numpy())
                        #    y1 = max(b_coordinates[1].cpu().numpy(), o_coordinates[1].cpu().numpy())
                        #    x2 = min(b_coordinates[2].cpu().numpy(), o_coordinates[2].cpu().numpy())
                        #    y2 = min(b_coordinates[3].cpu().numpy(), o_coordinates[3].cpu().numpy())

                        # Calculate area of intersection
                        #    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
                        #    b_area = b_center[2].cpu().numpy() * b_center[3].cpu().numpy()
                        #    if intersection_area / b_area > 0.5:
                        #        too_much_overlap = True
                        #        break
                        if not too_much_overlap:
                            confidence = result_bbox.conf
                            if confidence[0].cpu().numpy() > 0:
                                a = True
                                x, y, w, h = b_center.cpu().numpy()
                                dist = (x - self.obj[0])**2 + \
                                    (y - self.obj[1])**2
                                if dist <= min_dist:
                                    min_dist = dist
                                    best_x = x
                                    best_y = y
                                if (self.record or self.show):
                                    frame = cv2.circle(frame, (int(b_center[0].cpu().numpy()), int(
                                        b_center[1].cpu().numpy())), 5, (0, 0, 255), -1)
                                    frame = cv2.rectangle(
                                        frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                                    label = "castana"
                                    frame = cv2.putText(
                                        frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                break
                self.x = best_x
                self.y = best_y
                if (self.record):
                    # draw a triange of angle 7.5 from the midle bottom of the camera
                    # angle_radians = np.radians(90-7.5)
                    # direction_vector1 = np.array([np.cos(angle_radians), -np.sin(angle_radians)])
                    # direction_vector2 = np.array([-np.cos(angle_radians), -np.sin(angle_radians)])
                    # line_length = min(self.x_max, self.y_max)
                    # line1_points = intersection_point + line_length * direction_vector1
                    # line2_points = intersection_point + line_length * direction_vector2
                    # Convert points to integers
                    # line1_points = tuple(map(int, line1_points))
                    # line2_points = tuple(map(int, line2_points))
                    # print("a")
                    # print(line1_points, line2_points)
                    # Draw lines on the frame
                    frame = cv2.line(frame, (int(
                        self.x_max / 2) - 70, int(self.y_max)), (int(self.x_max / 2) - 70, 0), (255, 0, 0), 2)
                    frame = cv2.line(frame, (int(
                        self.x_max / 2) + 70, int(self.y_max)), (int(self.x_max / 2) + 70, 0), (255, 0, 0), 2)
                    out.write(frame)
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
        print("Starting...")
        self.tracker.initiateVideo()
        print("Video Started...")
        self.tracking_thread.start()
        print("Track thread Started...")
       
   

       

    def finish(self):
        # Close the serial port
        # Release the video writer after the main loop
        out.release()
        self.tracker.stop_tracking()
        self.tracker.finish()
        self.tracking_thread.join()


print("loading model...")

model = YOLO("best_f.pt")
print("model loaded")
brain = Brain(NutsTracker(model), Communication())
