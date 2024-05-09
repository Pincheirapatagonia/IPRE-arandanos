import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO
from threading import Lock, Thread
from time import sleep
import matplotlib.pyplot as plt
import scipy
#import ogl_viewer.viewer as gl

import cv_viewer.tracking_viewer as cv_viewer


# V1.0: NO SE TOMARA EN CONSIDERACION LAS MEDICIONES DE DISTANCIA DE LA CAMARA
# ZED.

# ------------------- Auxiliary Functions ------------------------------------

def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2), dtype=int)

    # Center / Width / Height -> BBox corners coordinates
    x_min = int(xywh[0] - 0.5*xywh[2]) #* im_shape[1]
    x_max = int(xywh[0] + 0.5*xywh[2]) #* im_shape[1]
    y_min = int(xywh[1] - 0.5*xywh[3]) #* im_shape[0]
    y_max = int(xywh[1] + 0.5*xywh[3]) #* im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output

def detections_to_custom_box(detections, im0):
    # [WARNING]: SE PUSO EN 0 OBJ LABEL Y PROBABILITY PARA QUE NO TIRARA
    # WARNINGS EN LA TERMINAL, ESTO SE DEBE CORREGIR!
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
        obj.label = 0 #det.cls
        obj.probability = 0 # det.conf
        obj.is_grounded = False
        output.append(obj)
    return output

# -----------------------------------------------------------------------------


class StereoCamera():
    
    def __init__(self, YOLO_file_path = 'ArandanosV2.pt', inference_pixel_size = 416,
                 confidence_thresh = 0.4, svo_file = None, iou_thresh = 0.45):
        
        self.confidence_thresh = confidence_thresh
        self.inference_pixel_size = inference_pixel_size
        self.svo_file = svo_file
        self.iou_thresh = iou_thresh
        self.YOLO_file_path = YOLO_file_path
        self.zed = sl.Camera()
        self.image_tmp_left = sl.Mat()
        self.image_tmp_right = sl.Mat()
        
        self.init_parms = None
        self.focal_x_left = None
        self.focal_y_left = None
        self.cx_left = None
        self.cy_left = None
        self.camera_baseline = None
        self.runtime_params = None
        
        self.viewer = None
        self.image_scale = None
        self.ocv_display = None
        self.image_ocv_track = None
        self.image_display = None
        
        self.YOLO_model = None
        self.YOLO_detections_left = None
        self.YOLO_detections_right = None
        
        self.cam_w_pose = sl.Pose()
        self.objects = sl.Objects()
        self.obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        
        self.exit_signal = False
        self.run_signal = False
        self.thread_lock = Lock()
        self.image_net_left = None
        self.image_net_right = None
        
        self.initialize_camera()
        self.initialize_viewer()
        self.initialize_YOLO()
    
    
    def initialize_camera(self):
        # Initialize camera settings 
        input_type = sl.InputType()
        
        if self.svo_file is not None:
            input_type.set_from_svo_file(self.svo_file)

        # Create a InitParameters object and set configuration parameters
        self.init_params = sl.InitParameters(input_t=input_type,
                                        svo_real_time_mode=True)

        status = self.zed.open(self.init_params)

        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            raise Exception
            
        calibration_params = self.zed.get_camera_information(
            ).camera_configuration.calibration_parameters
        self.focal_x_left = calibration_params.left_cam.fx
        self.focal_y_left = calibration_params.left_cam.fy
        self.cx_left = calibration_params.left_cam.cx
        self.cy_left = calibration_params.left_cam.cy
        
        
        
        self.camera_baseline = calibration_params.get_camera_baseline() # ESTA EN MILIMETROS
        
        self.runtime_params = sl.RuntimeParameters()
        
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        self.zed.enable_positional_tracking(positional_tracking_parameters)
        
        obj_param = sl.ObjectDetectionParameters()
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        obj_param.enable_tracking = True
        self.zed.enable_object_detection(obj_param)
        
        
        print("Camera initialized sucessfully...")
        
        
    def initialize_viewer(self):
        camera_infos = self.zed.get_camera_information()
        camera_res = camera_infos.camera_configuration.resolution
        
        # AL PARECER PARA MOSTRAR 2D NO ES NECESARIO EL GL VIEWER
        #self.viewer = gl.GLViewer()
        
        self.display_resolution = sl.Resolution(min(camera_res.width, 1280),
                                           min(camera_res.height, 720))
        self.image_scale = [self.display_resolution.width / camera_res.width,
                            self.display_resolution.height / camera_res.height]
        self.image_ocv_display = np.full((self.display_resolution.height,
                                       self.display_resolution.width, 4),
                                      [245, 239, 239, 255], np.uint8)
        self.image_display = sl.Mat()
        
        
        
        camera_config = camera_infos.camera_configuration
        tracks_resolution = sl.Resolution(400, self.display_resolution.height)
        self.track_view_generator = cv_viewer.TrackingViewer(tracks_resolution,
                                                        camera_config.fps,
                                                        self.init_params.depth_maximum_distance)
        self.track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
        self.image_ocv_track = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
        
        
        
        
        print("Viewer initialized successfully...")
    
    def initialize_YOLO(self):
        self.YOLO_model = YOLO(self.YOLO_file_path)
        print("YOLO MODEL initialized sucessfully...")
        
    
    def YOLO_thread(self):

        while not self.exit_signal:
            if self.run_signal:
                self.thread_lock.acquire()
                
                # YOLO detections on the left image
                img_left = cv2.cvtColor(self.image_net_left, cv2.COLOR_BGRA2BGR)
                det = self.YOLO_model.predict(img_left, save = False,
                                    imgsz = self.inference_pixel_size,
                                    conf = self.confidence_thresh,
                                    iou = self.iou_thresh,
                                    verbose = False)[0].cpu().numpy().boxes
                
                self.YOLO_detections_left = detections_to_custom_box(det, self.image_net_left)
                
                # YOLO detections on the right image
                img_right = cv2.cvtColor(self.image_net_right, cv2.COLOR_BGRA2BGR)
                det = self.YOLO_model.predict(img_right, save = False,
                                    imgsz = self.inference_pixel_size,
                                    conf = self.confidence_thresh,
                                    iou = self.iou_thresh,
                                    verbose = False)[0].cpu().numpy().boxes
                
                self.YOLO_detections_right = detections_to_custom_box(det, self.image_net_right)
                
                
                self.thread_lock.release()
                self.run_signal = False
            sleep(0.01)
            
            
    def run(self):
        
        print("Starting detection thread...")
        capture_thread = Thread(target=self.YOLO_thread)
        capture_thread.start()
        
        # while self.viewer.is_available() and not self.exit_signal: [OLD]
        while not self.exit_signal:
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                
                self.thread_lock.acquire()
                # -- Get the left image
                self.zed.retrieve_image(self.image_tmp_left, sl.VIEW.LEFT)
                self.image_net_left = self.image_tmp_left.get_data()
                
                # -- Get the right image 
                
                self.zed.retrieve_image(self.image_tmp_right, sl.VIEW.RIGHT)
                self.image_net_right = self.image_tmp_right.get_data()
                
                self.thread_lock.release()
                self.run_signal = True

                # -- Detection running on the other thread
                while self.run_signal:
                    sleep(0.001)

                # Wait for detections
                self.thread_lock.acquire()
                
                # Make a distance estimation to target
                self.target_pos = self.stereo_measure()
                
                # -- Ingest detections
                self.zed.ingest_custom_box_objects(self.YOLO_detections_left)
                self.thread_lock.release()
                self.zed.retrieve_objects(self.objects, self.obj_runtime_param)

                # -- Display
                # Retrieve display data
                self.zed.retrieve_image(self.image_display, sl.VIEW.LEFT,
                                        sl.MEM.CPU, self.display_resolution)

                # 2D rendering
                np.copyto(self.image_ocv_display, self.image_display.get_data())
                
                cv_viewer.render_2D(self.image_ocv_display,
                                    self.image_scale, self.objects,
                                    True)
                global_image = cv2.hconcat([self.image_ocv_display,
                                            self.image_ocv_track])
                # Tracking view
                self.track_view_generator.generate_view(self.objects,
                                                        self.cam_w_pose,
                                                        self.image_ocv_track,
                                                        self.objects.is_tracked)

                cv2.imshow("ZED | 2D View", global_image)
                
                key = cv2.waitKey(1)
                if key == 27:
                   self.exit_signal = True
            else:
                print("Error!, viewer is not available!")
                self.exit_signal = True

        #self.viewer.exit()
        self.exit_signal = True
        self.zed.close()
        
    
    # V2.0 se entrega la distancia como una medida x, y, z
    # algoritmo de deteccion es simple CROSSRELATION 
    # PELIGRO EN CASO DE ROTACIONES, RESIZES O DIFERENCIAS BRUSCAS DE ILUMINACION
    # ENTRE CAMARAS
    def stereo_measure(self):
        
        # V1.0 solo obtendra una medida si ambas detecciones son 1 bbox
        # y la disparidad sera entre centros de bbox
        if len(self.YOLO_detections_left) == len(self.YOLO_detections_right) == 1:
            

            center_x_left = (self.YOLO_detections_left[0].bounding_box_2d[0][0] + \
                      self.YOLO_detections_left[0].bounding_box_2d[2][0])/2
                
            center_y_left = (self.YOLO_detections_left[0].bounding_box_2d[0][1] + \
                      self.YOLO_detections_left[0].bounding_box_2d[2][1])/2
                
            center_x_right = (self.YOLO_detections_right[0].bounding_box_2d[0][0] + \
                      self.YOLO_detections_right[0].bounding_box_2d[2][0])/2
                
                
                
            bbox_left = self.image_net_left[int(self.YOLO_detections_left[0].bounding_box_2d[0][1]):
                                            int(self.YOLO_detections_left[0].bounding_box_2d[2][1]),
                                            int(self.YOLO_detections_left[0].bounding_box_2d[0][0]):
                                            int(self.YOLO_detections_left[0].bounding_box_2d[2][0])]
                
            
            bbox_right = self.image_net_right[int(self.YOLO_detections_right[0].bounding_box_2d[0][1]):
                                            int(self.YOLO_detections_right[0].bounding_box_2d[2][1]),
                                            int(self.YOLO_detections_right[0].bounding_box_2d[0][0]):
                                           int( self.YOLO_detections_right[0].bounding_box_2d[2][0])]
                
            # V1: disparidad como simple resta entre bbox
            dx = abs(center_x_left - center_x_right)
            
            # V2: se intenta corregir los bbox
            dx_correction = self.image_shift(bbox_left, bbox_right)
            
            disparity = dx + dx_correction
            
            target_z_aux = (self.focal_x_left *
                             self.camera_baseline)/dx
            target_z = (self.focal_x_left * self.camera_baseline)/disparity
            
            target_x = (center_x_left - self.cx_left) * (target_z/self.focal_x_left)
            target_y = (center_y_left - self.cy_left) * (target_z/self.focal_y_left)
            
            print("Disparity is: " + str(dx))
            print("Disparity correction: " + str(dx_correction))
            print("Blueberry position: ({}, {}, {})".format(
                 target_x, target_y, target_z))
            print(f"Blueberry position: ({target_x}, {target_y}, {target_z})\n")
            print(f"Blueberry aux position: ({target_x}, {target_y}, {target_z_aux})")
            return (target_x, target_y, target_z)
    
    
    def image_shift(self, bbox1, bbox2):
        bbox1_gray = np.sum(bbox1.astype('float'), axis=2)
        bbox1_gray -= np.mean(bbox1_gray)
        bbox2_gray = np.sum(bbox2.astype('float'), axis=2)
        bbox2_gray -= np.mean(bbox1_gray)
        
        corr_out = scipy.signal.correlate(bbox1_gray, bbox2_gray, mode="same")

        x_dis = np.unravel_index(np.argmax(corr_out), bbox1_gray.shape)[1] - bbox1_gray.shape[1]/2
        return x_dis
        

        
        
        
        
            
    
if __name__ == '__main__':
    YOLO_file_path = 'ArandanosV2.pt'
    confidence_thresh = 0.1
    
    
    
    stereo_camera = StereoCamera(YOLO_file_path = YOLO_file_path,
                                 confidence_thresh = confidence_thresh)
    stereo_camera.run()
    