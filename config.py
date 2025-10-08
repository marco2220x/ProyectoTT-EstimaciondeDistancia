# Contains initialization of necessary variables, models, and etc coming from the config.yml file

import yaml
import cv2

# Load the YAML config file
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# Load camera information
sensor_height_mm = config["camera_information"]["sensor_height_mm"] 
sensor_height_px = config["camera_information"]["sensor_height_px"] 
focal_length = config["camera_information"]["focal_length"]

# Load target object information
real_object_height = config["target_object"]["real_object_height"]
target = config["target_object"]["target"]
