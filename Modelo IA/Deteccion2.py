import math
import cv2
import numpy as np
import serial
import time
import keyboard
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
        self.distancia = "0"
        self.area = "0"
        self.tracking = True
        self.show = True
        self.x = -1
        self.y = -1
        self.x_max = 0
        self.y_max = 0
        self.model = model
        self.obj = [0, 0]
        self.camera_num = 0

    def initiateVideo(self):
        print("Trying to open camera...")
        self.cap = cv2.VideoCapture(self.camera_num)
        while not self.cap.isOpened():
            print("Error opening video")
            self.cap = cv2.VideoCapture(self.camera_num)
        print("Camera opened.")
        ret, frame = self.cap.read()
        while (not ret):
            print(ret, frame)
            ret, frame = self.cap.read()
        self.y_max, self.x_max, _ = frame.shape
        self.obj = [int(self.x_max / 2), int(self.y_max)]

    def track(self):
        print("Tracking...")
        while self.tracking:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            # Make predictions
            with torch.no_grad():
                predictions = self.model(frame)
                print(predictions)
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
                       
                        if not too_much_overlap:
                            confidence = result_bbox.conf
                            if confidence[0].cpu().numpy() > 0:
                
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
                                    label = "Arandanos"
                                    frame = cv2.putText(
                                        frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                break
                self.x = best_x
                self.y = best_y
                if (self.record):
                    out.write(frame)
            if self.show:
                cv2.imshow('Object Detection', frame)
                cv2.waitKey(1)

    def stop_tracking(self):
        self.tracking = False

    def finish(self):
        self.cap.release()
        cv2.destroyAllWindows()


class Communication:
    def __init__(self) -> None:
        self.mostrar_contorno = False
        self.manual_mode = False
        self.starts = False
        self.target_W = "COM7"
        self.target_L = '/dev/ttyACM0'
        self.baud = 9600
        self.data = ''
        self.messages = True
        self.on = False

    def begin(self):
        if self.on:
            self.device = serial.Serial(self.target_W, self.baud, timeout=1)
            time.sleep(0.1)
            if self.device.isOpen():
                print("{} connected!".format(self.device.port))
                time.sleep(1)

    def read_and_print_messages(self):
        if self.on:
            while self.messages:
                try:
                    if self.device.isOpen():
                        message = self.device.readline().decode('utf-8').strip()
                        if message:
                            self.data = message
                except Exception as e:
                    print(f"Error reading message: {e}")

    def comunicacion(self, mensaje):
        if self.on and self.device.isOpen():
            self.device.flush()
            self.device.write(mensaje.encode('utf-8'))
            time.sleep(0.1)

    def stop_messages(self):
        if self.on:
            self.messages = False


class Brain:
    def __init__(self, tracker, coms) -> None:
        
        self.tracker = tracker
        self.coms = coms
        
        print("Starting threads...")
        self.tracking_thread = threading.Thread(
            target=self.track_wrapper, args=())
        self.tracking_thread.daemon = True

        if self.coms.on:
            self.read_messages_thread = threading.Thread(
                target=self.coms.read_and_print_messages)
            self.read_messages_thread.daemon = True

        
        self.distance = 1
        
        self.begin()
        self.do()

    def track_wrapper(self):
        # This function is used to run the track() method in a separate thread.
        print("Tracking thread started...")
        self.tracker.track()

    def begin(self):
        print("Starting...")
        self.tracker.initiateVideo()
        print("Video Started...")
        
        if self.coms.on:
            self.coms.begin()
            print("Coms Started...")
            self.tracking_thread.start()
            print("Track thread Started...")
            self.read_messages_thread.start()
            print("read message thread Started...")
        
        
    def do(self):
        running = True
        try:
            while running: # Loop until the user presses 'q'
                self.track_wrapper()        
        except KeyboardInterrupt:
            print("Data collection interrupted.")

        finally:
            self.finish()


    def finish(self):
        # Close the serial port
        # Release the video writer after the main loop
        out.release()
        if self.coms.on:
            
            self.coms.device.close()
            self.coms.stop_messages()
            self.read_messages_thread.join()
            
        self.tracker.stop_tracking()
        self.tracker.finish()
        self.tracking_thread.join()


print("loading model...")

model = YOLO("ArandanosV1.pt")
print("model loaded")
brain = Brain(Tracker(model), Communication())
