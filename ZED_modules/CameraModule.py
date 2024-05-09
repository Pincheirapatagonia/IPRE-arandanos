import pyzed.sl as sl
import numpy as np

class CameraModule:
    
    
    def __init__(self):
        # ---- Parametros iniciales de la camara ---
        self.init_params = sl.InitParameters()
        
        # Resolucion / FPS
        # HD2K -> 15 FPS
        # HD1080 -> 30/15 FPS
        # HD720 -> 60/30/15 FPS
        # VGA -> 100/60/30/15 FPS
        self.init_params.camera_resolution = sl.RESOLUTION.HD720 
        self.init_params.camera_fps = 15
        
        # Definicion de coordenadas
        # z hacia arriba, x a lo largo de la camara, y: eje perpendicular al lente
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER
        
        # Definicion de parametros de profundidad
        # none: sin profundidad, neural: profundidad de calidad con mucho uso de gpu
        # NONE/PERFORMANCE/QUALITY/ULTRA/NEURAL 
        self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        
        # ---- Parametros de Runtime (llamado cuando se toma una nueva foto)
        # de la camara ----
        self.runtime_parameters = sl.RuntimeParameters() 
        
        # Configurar el rango de aceptacion de la medicion de profundidad (que
        # tan bien se hace el calce de imagenes / que tan bien calzan las texturas)
        # el rango es de 0 - 100, 100 indica que toda medicion es aceptada, por lo
        # que el mapa de profundidades es muy denso. 10 indica que solo las
        # mediciones de profundidad que la camara esta totalmente seguras son
        # tomadas en cuenta (pero la mayoria de la imagen se vera en negro)
        
        # [COMO ES CALCULADA LA CONFIANZA?]
        self.runtime_parameters.confidence_threshold = 100
        self.runtime_parameters.textureness_confidence_threshold = 100
        
        # Elegir el frame referencial, es decir, si el origen de coordenadas es
        # la pose inicial de la camara o lo definido como "WORLD":
        # CAMERA/WORLD
        self.runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
        
        # ---- Parametros de tracking de posicionamiento (mediante odometria
        # optica) ----
        # [IMPORTANTE]: La camara retorna solo la posicion o la aceleracion
        # pero NO la velocidad
        
        # Transformacion vacia? (asume que la camara es el origen?)
        py_transform = sl.Transform()  
        self.tracking_parameters = sl.PositionalTrackingParameters(
            _init_pos=py_transform)
        
        # Smoothing de posicionamiento, corregir leves drifts 
        self.tracking_parameters.enable_pose_smoothing = True
        
        # Activar IMU y usarlo en conjunto de la odometria optica para calcular
        # la posicion de la camara
        self.tracking_parameters.enable_imu_fusion = True
        
        # ---- Parametros de mapping ----
        self.mapping_parameters = sl.SpatialMappingParameters()
        
        # Resolucion, para outdoor se recomienda poca resolucion ya que pide mucha
        # memoria: HIGH/MEDIUM/LOW
        self.mapping_parameters.set_resolution(sl.MAPPING_RESOLUTION.LOW)
        
        # Rango de mapeo (que tan lejos puede medir e incorporar al mapa):
        # SHORT/MEDIUM/LONG/AUTO
        self.mapping_parameters.set_range(sl.MAPPING_RANGE.MEDIUM)
        
        # Maximo uso de memoria del CPU en megabytes que se puede destinar al mapa
        self.mapping_parameters.max_memory_usage = 2048
        
        # Si se desea aplicar textura al mapa (consume mas memoria)
        self.mapping_parameters.save_texture = False
        
        # consistencia del mesh con los datos internos: True -> mas rapido
        # False -> mejor mapa
        self.mapping_parameters.use_chunk_only = False
        
        # Tipo de mapa, los mesh son poligonos 3D que no tienen informacion del
        # color, mientras que la nube de puntos son vertices inconexos con 
        # informacion de color:  FUSED_POINT_CLOUD / MESH

        
        # --- Parametros iniciales de deteccion de objetos --- 
        self.detection_parameters = sl.ObjectDetectionParameters()
        
        # Tipos de modelos de deteccion estan hechos para detectar humanos u
        # objetos, tambien da la opcion de poner el modelo proprio
        # MULTI_CLASS_BOX / MULTI_CLASS_BOX_ACCURATE / HUMAN_BODY_FAST
        # MULTI_CLASS_BOX_MEDIUM / CUSTOM_BOX_OBJECT
        self.detection_parameters.detection_model = sl.DETECTION_MODEL.MULTI_CLASS_BOX
        
        # Filtro para eliminar clases que ya han sido trackeadas antes y se 
        # encuentran en la misma posicion 3D.   
        # NONE / NMS3D / NMS3D_PER_CLASS
        self.detection_parameters.filtering_mode = sl.OBJECT_FILTERING_MODE.NONE
        
        # Sincronizacion de obtencion de imagenes: si es que se desea que 
        # la camara retorne la imagen y los objetos a la vez o en distintos
        # threads: False / True
        self.detection_parameters.image_sync = True
        
        # Trackeo de objetos, si es que se desea que la camara siga el 
        # movimiento de los objetos detectados durante las distintas imagenes:
        # False / True
        self.detection_parameters.enable_tracking = False
        
        # Mask Obj?
        self.detection_parameters.enable_mask_output = False
        
        # --- Parametros Runtime de deteccion de objetos --- 
        self.run_detection_parameters = sl.ObjectDetectionRuntimeParameters()
        
        # Threshold de confianza de los objetos detectados (1 - 99)
        # 1 -> practicamente se aceptan todos los bbox.
        # 99 -> solo quedan los bbox que la IA esta completamente segura
        self.run_detection_parameters.detection_confidence_threshold = 1
        
        # filtro de clases (que tipo de clase se tratara de detectar)
        # PERSON / VEHICLE / BAG / ANIMAL / ELECTRONICS / FRUIT_VEGETABLE / 
        # SPORT / [] (lista vacia indica que se detectan todas las clases
        self.run_detection_parameters.object_class_filter = [
            sl.OBJECT_CLASS.FRUIT_VEGETABLE]
        
    def start(self):
        # [IMPORTANTE]: los siguientes sensores de la camara no son utilizados 
        # (barometro / temperatura / magnetometro).
        # El IMU es usado en conjunto con la odometria optica para obtener la
        # posicion (guardado en cam_pose) y tambien es usado para conocer la
        # aceleracion lineal.
        self.camera = sl.Camera()
        
        # Se setean los parametros iniciales, en caso de un error se caera
        # el programa.
        err = self.camera.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(err)
            raise Exception
        
        # Se setean los parametros de posicionamiento
        err = self.camera.enable_positional_tracking(self.tracking_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            print(err)
            raise Exception
        
        # Se setean los parametros de mapeo
        err = self.camera.enable_spatial_mapping(self.mapping_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            print(err)
            raise Exception
            
        # Se setean los parametros iniciales de deteccion de objetos
        err = self.camera.enable_object_detection(self.detection_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            print(err)
            raise Exception
        
        print("Camera initialized!")
        
        # Se instancian las variables que guardaran las clases de los datos
        # obtenidos por la camara.
        self._cam_img = sl.Mat()
        self._cam_depth = sl.Mat() 
        self._cam_depth_xyz = sl.Mat() 
        self._cam_pose = sl.Pose()
        self._cam_mesh = sl.Mesh()
        self._cam_sensors = sl.SensorsData()
        self._cam_objects = sl.Objects()
        self._cam_time = sl.Timestamp()
        
        # Se instancian las variables que guardan los datos extraidos de las 
        # clases.
        self.cam_img = np.array([])
        self.cam_depth = np.array([])
        self.cam_depth_xyz = np.array([])
        self.cam_pose = np.array([])
        self.cam_acc = np.array([])
        self.cam_obj_bbx = []
        self.cam_time = 0
    
    def get_data(self, show = False):
        # Funcion para ser llamada dentro de un loop
        # aolo existe la opcion sl.MEM.CPU por ahora.
        sucess = False
        if self.camera.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # --- Se guardan los datos en las clases respectivas --- 
            self.camera.retrieve_image(self._cam_img) # left por default
            self.camera.retrieve_measure(self._cam_depth, sl.MEASURE.DEPTH)
            self.camera.retrieve_measure(self._cam_depth_xyz, sl.MEASURE.XYZ)
            # notar que aunque se setea a "WORLD" este retorna la posicion
            # con respecto al origen (que puede ser la pos inicial de la 
            # camara).
            self.camera.get_position(self._cam_pose, sl.REFERENCE_FRAME.WORLD)
            self.camera.get_sensors_data(self._cam_sensors, sl.TIME_REFERENCE.IMAGE)
            self._cam_time = self.camera.get_timestamp(sl.TIME_REFERENCE.CURRENT)
            self.camera.retrieve_objects(self._cam_objects, self.run_detection_parameters)
            
            # Actualiza el mapa mesh
            _map_state = self.camera.get_spatial_mapping_state()
            if _map_state == sl.SPATIAL_MAPPING_STATE.OK:
                map_state = True
            else:
                map_state = False
            
            # --- Guardar los datos relevantes en las variables instanciadas ---
            self.cam_img = self._cam_img.get_data()
            # -inf "muy cera", +inf "muy lejos", nan "invalido/oclusion"
            # Se reemplazara los nan por "muy lejos" (quizas sea necesario
            # cambiar eso) 
            self.cam_depth = self._cam_depth.get_data()
            np.nan_to_num(self.cam_depth, copy=False, nan=20000, posinf=20000,
                          neginf=200)
            self.cam_depth_xyz = self._cam_depth_xyz.get_data()[:, :, 0:-1]
            np.nan_to_num(self.cam_depth_xyz, copy=False, nan=20000, posinf=20000,
                          neginf=200)
            # posicion xyz de la camara
            self.cam_pose = self._cam_pose.get_translation().get()
            # aceleracion de la camara
            self.cam_acc = self._cam_sensors.get_imu_data().get_linear_acceleration()
            # retornar objetos detectados
            self.cam_obj_bbx = self._cam_objects.object_list
            # retornar tiempo
            self.cam_time = self._cam_time.get_seconds()
            
            sucess = True
        return (sucess, self.cam_img, self.cam_depth, self.cam_depth_xyz,
                self.cam_pose, self.cam_acc, map_state, self.cam_obj_bbx,
                self.cam_time)
    
    def close(self):
        # se libera la memoria (se cae el programa?!)
        #self._cam_img.free()
        #self._cam_depth.free()
        #self._cam_depth_xyz.free() 
        
        # se desabilitan los modulos
        self.camera.disable_spatial_mapping()
        self.camera.disable_positional_tracking()
        self.camera.disable_object_detection()
        self.camera.close()