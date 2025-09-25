# modules/TargetObjectLocator.py
import cv2
import numpy as np
from modules.LensOpticCalculator import LensOpticCalculator, LimitVal

def FindTargetObject(img, target_bbox, mde_Model):
    """
    img: esperará la misma representación que usas en MonocularEstimator (en tu main conviertes a RGB antes de llamar, así que img = RGB)
    target_bbox: (x1, y1, x2, y2) en coordenadas del frame, o None
    mde_Model: matriz de profundidad retornada por MonocularEstimator (same size as img)
    """
    if target_bbox is None:
        return None

    img_copy = img.copy()
    img_height, img_width, _ = img_copy.shape

    x1, y1, x2, y2 = target_bbox
    # Asegurar límites
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_width - 1, x2); y2 = min(img_height - 1, y2)

    # ROI del target
    img_roi = img_copy[y1:y2, x1:x2]

    # Dibujar bbox en la imagen principal (color negro como en tu original)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

    # Centro del bbox (coordenadas en el frame completo)
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)

    xoffset_coord = LimitVal(x_center, img_width)
    yoffset_coord = LimitVal(y_center, img_height)

    # Dibujar centro
    cv2.circle(img, (xoffset_coord, yoffset_coord), 3, (0, 0, 0), 2)

    # Obtener valor de profundidad del MDE en el centro
    target_midas_val = mde_Model[yoffset_coord, xoffset_coord]

    # Usar la altura del bbox para el cálculo de lente (igual que antes)
    height = y2 - y1
    target_computed_depthmap_val = LensOpticCalculator(height)

    # Mostrar ROI (convertir si tu img es RGB -> BGR para mostrar)
    try:
        cv2.imshow('ROI', cv2.cvtColor(img_roi, cv2.COLOR_RGB2BGR))
    except:
        # si img_roi ya está en BGR o falla la conversión, mostrar directo
        cv2.imshow('ROI', img_roi)

    return (target_computed_depthmap_val, target_midas_val)
