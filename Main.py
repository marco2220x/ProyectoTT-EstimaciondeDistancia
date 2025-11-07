# main_dataset.py (versión extendida con features geométricas y normalizadas)
import cv2
import os
import csv
import math
from glob import glob
from ultralytics import YOLO
from config import sensor_width_mm, sensor_width_px, focal_length, real_object_height

from modules.ROITarget import TargetTracker
from modules.LensOpticCalculator import LensOpticCalculator

# --- Configuración ---
input_dir = "test"
step_min = 6
iou_threshold = 0.85
output_dir = "dataset"
images_dir = os.path.join(output_dir, "images")
os.makedirs(images_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "new_data.csv")
csv_exists = os.path.exists(csv_path)

# --- Inicializar YOLO + Tracker ---
yolo = YOLO("yolov8n.pt")
tracker = TargetTracker(yolo_model=yolo)

# --- Parámetros de cámara ---
focal_length_px = focal_length * sensor_width_px / sensor_width_mm
real_height_m = real_object_height / 1000.0
target_id = 5 # cambiar el target de cada persona

# --- Función IoU ---
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


# --- Abrir CSV ---
with open(csv_path, mode="a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if not csv_exists:
        # Encabezado actualizado con nuevas features
        writer.writerow([
            "filename",
            "xmin", "ymin", "xmax", "ymax",
            "width", "height",
            "distance",
            "x_center", "y_center",
            "real_height_m", "height_rel", 
            "target_id"
        ])

    saved_idx = 0
    video_paths = sorted(glob(os.path.join(input_dir, "vid_28.mp4")))

    for input_path in video_paths:
        print(f"[INFO] Procesando video: {input_path}")
        basename = os.path.splitext(os.path.basename(input_path))[0]

        cap = cv2.VideoCapture(input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Resolución del video: {frame_width}x{frame_height}")

        prev_bbox = None
        frames_since_last_save = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_since_last_save += 1
            target_bbox, _ = tracker.update(frame, draw=False)
            if target_bbox is None:
                continue

            save_frame = True
            if prev_bbox is not None:
                iou = compute_iou(prev_bbox, target_bbox)
                if iou > iou_threshold and frames_since_last_save < step_min:
                    save_frame = False

            if not save_frame:
                continue

            prev_bbox = target_bbox
            frames_since_last_save = 0
            x1, y1, x2, y2 = target_bbox

            # --- Calcular dimensiones ---
            width = x2 - x1
            height = y2 - y1
            if height <= 0 or width <= 0:
                continue

            # --- Calcular distancia ---
            distance_m = LensOpticCalculator(height) / 1000.0  # mm → m

            # --- Calcular centro ---
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0

            # --- Guardar imagen ---
            filename = f"{basename}_{saved_idx:05d}.jpg"
            filepath = os.path.join(images_dir, filename)
            cv2.imwrite(filepath, frame)

            # === Calcular nuevas features ===
            height_rel = height / real_height_m
        
            # --- Escribir fila completa ---
            writer.writerow([
                filename,
                f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                f"{width:.1f}", f"{height:.1f}",
                f"{distance_m:.3f}",
                f"{x_center:.1f}", f"{y_center:.1f}",
                f"{real_height_m:.3f}", f"{height_rel:.3f}"
                f"{target_id}",

                
            ])
            saved_idx += 1

        cap.release()
        print(f"[INFO] Video {basename} procesado, frames guardados: {saved_idx}")

print(f"[INFO] Dataset completo generado en {output_dir}")
