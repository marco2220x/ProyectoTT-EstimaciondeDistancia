# Contains the function that generates the Depth Map of the input feed.
# Cambios a implementar: Probar con otro modelo de estimación de profundidad.

import cv2
from config import mde_model

def MonocularEstimator(img):
    img_height, img_width, channels = img.shape                
    blob = cv2.dnn.blobFromImage(img,1/255.,(256,256),(123.675, 116.28, 103.53), True, False)    #Qué hace blob?                              # Represents the feed into a blob network.
    # El mapa de profundidad generado por MiDas originalmente visualiza con un mapa de calor ?
    # Se pasa blob al modelo de estimación de profundidad                                                                                                                           # When using the large model change the blob size to (384,384) instead of (256,256)3.                             
    mde_model.setInput(blob)                                                                                                   # Inputs the blob to the MiDaS MDE Model.
    monocular_output = mde_model.forward()                  
    monocular_output = monocular_output [0,:,:] # Cuál es la primera capa del output?
    monocular_output = cv2.resize(monocular_output,(img_width, img_height))  # Se reescala la salida al tamaño original del video                                             # Resizes the MDE Feed similar to the input video feed.
    monocular_output = cv2.normalize(monocular_output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)   # Los valores de la profundidad estan en un rango de 0-1, porqué de esta forma y no usando el mapa de calor "crudo"           # Normalized the MDE Values in a range from 0-1. E
    return monocular_output                                                                                                    # Returns the MDE feed array.