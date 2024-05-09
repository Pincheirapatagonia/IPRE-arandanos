import numpy as np
import time
from ultralytics import YOLO
import torch
import threading
import cv2


class BlueberryTracker:
    def __init__(self, resolution=(2028, 1520), framerate=12,):
        self.record = True
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.resolution = resolution
        self.framerate = framerate
        self.frame = None
        self.stopped = False
        self.tracking = True
        self.show = True
        self.mostrar_contorno = True
        self.x = -1
        self.y = -1
        self.x_max = resolution[0]
        self.y_max = resolution[1]
        self.detect = 0
        self.obj = [0, 0]
        self.min_area = 500
        self.max_area = 10000
        self.default_lower = np.array([29, 43, 161])
        self.default_upper = np.array([91, 255, 255])
        self.detect = 0
        self.objX = resolution[0]/2
        self.objY = resolution[1]
        self.box = [0, 0, 0, 0]
        self.Zref = 50 #cm
        self.Rref = 0
        self.R = 0
        self.Z = 0
        self.calibrating= True
        self.conf = 0.2
        
        
    def initiateVideo(self):
        print("Initiating video.")
        self.camera = cv2.VideoCapture(0)  # Set camera to built-in webcam
        ret, self.frame = self.camera.read()  # Capture a frame
        if not ret:
            print("Failed to grab frame")
            self.x_max = self.frame.shape[0]
            self.y_max = self.frame.shape[1]
            time.sleep(1)
            print("Video initiated.")
            
    def calibrate(self, model):
        while self.calibrating:
            ret, self.frame = self.camera.read()
            if not ret:
                break
            if self.show:
                self.predict_frames(model)
                self.frame = cv2.resize(self.frame, None, fx=2, fy=2)
                cv2.imshow("Frame", self.frame)
                if self.R != 0 and self.R != None:
                    self.Rref = self.R
                    print(f"Rref: {self.Rref}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                
                self.calibrating = False
                print(f"Calibration finished. Rref: {self.Rref}")
                cv2.destroyAllWindows()
                break
        
        
    def det_z(self):
        if self.R != 0 and self.R != None:
            self.Z = (self.Zref*self.Rref)/self.R
            
            print(f"Z: {self.Z}, R: {self.R}, Zref: {self.Zref}, Rref: {self.Rref}")
         
    def predict_frames(self, model):
        image = self.frame
        with torch.no_grad():
            predictions = model(image, verbose=False, conf=self.conf)
        for result in predictions:
            result_bboxes = result.boxes
            for result_bbox in result_bboxes:
                b_coordinates = result_bbox.xyxy[0]
                self.box = b_coordinates[:4].int().tolist()
                lower_hsv = np.array([100, 100, 100])
                upper_hsv = np.array([255, 255, 255])
                self.find_blue_object(lower_hsv, upper_hsv, self.box)
                    
    def find_blue_object(self, lower_hsv, upper_hsv, bounding_box):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        mask[0:bounding_box[1], :] = 0
        mask[bounding_box[3]:, :] = 0
        mask[:, 0:bounding_box[0]] = 0
        mask[:, bounding_box[2]:] = 0

        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area = None
        radius = None

        if contours:
            # Draw bounding box
            cv2.rectangle(self.frame, (bounding_box[0], bounding_box[1]), (
            bounding_box[2], bounding_box[3]), (0, 255, 0), 2)
            max_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(max_contour)
            center = (int(x), int(y))
            radius = int(radius)

            cv2.circle(self.frame, center, radius, (0, 255, 0), 2)
            cv2.drawContours(self.frame, [max_contour], -1, (0, 0, 255), 2)

            area = cv2.contourArea(max_contour)
            self.R = radius
            
            if self.Rref != 0 and self.Rref != None:
                self.det_z()
                
            cv2.putText(self.frame, f'Area: {area}, Radio: {radius}, Z: {np.round(self.Z)} cm',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if area != None and radius != None: 
            #print(f"Area: {area}, Diameter: {radius*2}, Z: {self.Z}cm")
            pass

    def do(self, model):
        while not self.stopped:
            ret, self.frame = self.camera.read()
            if not ret:
                break
            if self.show:
                self.predict_frames(model)
                self.frame = cv2.resize(self.frame, None, fx=2, fy=2)
                cv2.imshow("Frame", self.frame)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                self.tracking = False
                break
    def finish(self):
        print("Windows released.")
        self.stopped = True
        self.camera.release()
        
        if(self.show):
            cv2.destroyAllWindows()



if __name__ == "__main__":
    model = YOLO("ArandanosV2.pt")
    tracker = BlueberryTracker()
    tracker.initiateVideo()
    tracker.calibrate(model)
    if tracker.calibrating == False:
        print("Calibration finished.")
        tracker.do(model)
        tracker.finish()