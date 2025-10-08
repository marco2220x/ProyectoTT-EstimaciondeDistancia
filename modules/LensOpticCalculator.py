# Contains functions regarding Lens Optic Calculation, Ratio and Proportion Calutation, and other auxilairy functions invovlving an object's distance safety level and value limiter.

from config import sensor_height_mm, sensor_height_px, focal_length, real_object_height

# Formulation of the lens Optic Calculation based on the Thin Lens Model.
def LensOpticCalculator (px_height):
    computed_object_distance = (real_object_height*focal_length*sensor_height_px)/(px_height*sensor_height_mm)                                
    return computed_object_distance
