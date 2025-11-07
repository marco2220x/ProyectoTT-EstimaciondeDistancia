# main_vid10_processed_corrected.py
import cv2
import os
import numpy as np
import time

from ultralytics import YOLO
from modules.ROITarget import TargetTracker
from modules.LensOpticCalculator import LensOpticCalculator

# === CONFIGURACIÓN ===
input_path = "test/vid_28.mp4"
output_dir = "runs_test"
os.makedirs(output_dir, exist_ok=True)

basename = os.path.splitext(os.path.basename(input_path))[0]
output_path = os.path.join(output_dir, f"{basename}_rect.mp4")

# === ABRIR VIDEO ===
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir el archivo de vídeo: {input_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# === INICIALIZAR YOLO y TRACKER ===
yolo_model = YOLO("yolov8n.pt")  # instancia de YOLO
tracker = TargetTracker(yolo_model=yolo_model,
                        max_age=30,
                        conf_thresh=0.35,
                        sim_thresh=0.7,
                        max_missed=60)

frame_idx = 0
start_time = time.time()
print(f"[INFO] Procesando {input_path} -> {output_path} (fps={fps}, size=({w},{h}))")

# === PROCESAMIENTO FRAME A FRAME ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    bgr_frame = frame
    out_frame = bgr_frame.copy()

    # 2. Actualizar tracker
    target_bbox, detections = tracker.update(bgr_frame, draw=False)

    distance_m = None
    if target_bbox is not None:
        x1, y1, x2, y2 = target_bbox
        # Centro del bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        # Altura en pixeles
        height_px = y2 - y1

        # Distancia estimada
        if height_px > 0:
            distance_m = LensOpticCalculator(height_px) / 1000.0  # mm -> m

        # Dibujar bbox y centro
        cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(out_frame, (cx, cy), 5, (0, 0, 255), -1)

        # === Mostrar información ===
        cv2.putText(out_frame, f"H: {height_px}px", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(out_frame, f"center_y: {cy}", (cx + 10, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
        if distance_m is not None:
            cv2.putText(out_frame, f"D: {distance_m:.2f} m", (cx + 10, cy + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25, 255, 255), 2)

    # 3. Guardar frame
    out.write(out_frame)

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"[INFO] Procesados {frame_idx} frames...")

# === FINALIZAR ===
cap.release()
out.release()
elapsed = time.time() - start_time
print(f"[INFO] {basename}: terminado ({frame_idx} frames, {elapsed:.1f}s).")
print(f"[INFO] Video procesado guardado en: {output_path}")
