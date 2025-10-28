# main_dataset.py (generar dataset de frames + CSV del target con IoU)
import cv2
import os
import csv
from glob import glob
from ultralytics import YOLO

from modules.ROITarget import TargetTracker
from modules.LensOpticCalculator import LensOpticCalculator

# --- Configuración ---
input_dir = "test"
step_min = 5  # mínimo número de frames a saltar
iou_threshold = 0.85  # IoU máximo para considerar frame similar
output_dir = "dataset"
images_dir = os.path.join(output_dir, "images")
os.makedirs(images_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "annotations.csv")
csv_exists = os.path.exists(csv_path)

# --- Inicializar YOLO + Tracker ---
yolo = YOLO("yolov8n.pt")
tracker = TargetTracker(yolo_model=yolo)

# --- Función IoU ---
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# --- Abrir CSV ---
with open(csv_path, mode="a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if not csv_exists:
        writer.writerow(["filename", "xmin", "ymin", "xmax", "ymax", 
                         "width", "height", "distance", 
                         "x_center", "y_center",])

    saved_idx = 0

    # --- Recorrer todos los videos ---
    video_paths = sorted(glob(os.path.join(input_dir, "*.mp4")))
    for input_path in video_paths:
        print(f"[INFO] Procesando video: {input_path}")
        basename = os.path.splitext(os.path.basename(input_path))[0]

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        prev_bbox = None
        frames_since_last_save = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            frames_since_last_save += 1

            target_bbox, _ = tracker.update(frame, draw=False)
            if target_bbox is None:
                continue

            # --- Calcular IoU con frame previo ---
            save_frame = True
            if prev_bbox is not None:
                iou = compute_iou(prev_bbox, target_bbox)
                if iou > iou_threshold and frames_since_last_save < step_min:
                    save_frame = False

            if not save_frame:
                continue

            # --- Guardar bbox anterior ---
            prev_bbox = target_bbox
            frames_since_last_save = 0

            x1, y1, x2, y2 = target_bbox

            width = x2 - x1
            height = y2 - y1
            if height <= 0 or width <= 0:
                continue

            # --- Calcular distancia solo con LensOpticCalculator ---
            distance_m = LensOpticCalculator(height) / 1000.0  # convertir mm -> m

            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            # --- Guardar frame como imagen ---
            filename = f"{basename}_{saved_idx:05d}.jpg"
            filepath = os.path.join(images_dir, filename)
            cv2.imwrite(filepath, frame)

            # --- Guardar fila en CSV ---
            writer.writerow([filename, x1, y1, x2, y2,  f"{width:.1f}", f"{height:.1f}",
                f"{distance_m:.3f}",
                f"{x_center:.1f}", f"{y_center:.1f}",])
            saved_idx += 1

        cap.release()
        print(f"[INFO] Video {basename} procesado, frames guardados: {saved_idx}")

print(f"[INFO] Dataset completo generado en {output_dir}")
