# Contains initialization of necessary variables, models, and etc coming from the config.yml file

import yaml
import cv2

# Load the YAML config file
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# Load model paths
coco_names = config["model_path"]["coco_names"]
coco_yolo_cfg = config["model_path"]["coco_yolo_cfg"]
coco_yolo_weights = config["model_path"]["coco_yolo_weights"]
MDE_model_path = config["model_path"]["MDE_model_path"]

# Load camera information
sensor_height_mm = config["camera_information"]["sensor_height_mm"] 
sensor_height_px = config["camera_information"]["sensor_height_px"] 
focal_length = config["camera_information"]["focal_length"]

# Load target object information
real_object_height = config["target_object"]["real_object_height"]
target = config["target_object"]["target"]

# Load class names (COCO)
class_names = []
with open(coco_names, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# Initialize YOLO model (COCO - detects people)
yolo_model = cv2.dnn.readNetFromDarknet(coco_yolo_cfg, coco_yolo_weights)
yolo_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Initialize the Monocular Depth Estimation (MDE) model
mde_model = cv2.dnn.readNet(MDE_model_path)
mde_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # CPU backend
mde_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)       # CPU target

# Checkpoints
print('YOLOv3 (COCO) Initialization Successful')
print('Monocular Depth Estimation Model Initialization Successful')
