# main.py (procesamiento OFFLINE -> guarda video resultante)
import cv2
import os
import numpy as np
import time

from modules.SceneLocator import FindObjects
from modules.MonocularEstimator import MonocularEstimator
from modules.TargetObjectLocator import FindTargetObject
from modules.ROITarget import TargetTracker

# --- Configuración de paths ---
input_path = "test/test2.mp4"
output_dir = "runs"
os.makedirs(output_dir, exist_ok=True)

basename = os.path.splitext(os.path.basename(input_path))[0]
output_path = os.path.join(output_dir, f"{basename}_processed.mp4")

# --- Abrir video ---
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir el archivo de vídeo: {input_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30  # fallback
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# --- Inicializar tracker (carga YOLO una vez) ---
tracker = TargetTracker(model_path="yolov8n.pt",
                        max_age=30,
                        conf_thresh=0.35,
                        sim_thresh=0.7,
                        max_missed=60)

frame_idx = 0
print(f"[INFO] Procesando {input_path} -> {output_path} (fps={fps}, size=({w},{h}))")

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    bgr_frame = frame  # VideoCapture devuelve BGR
    # Convertir a RGB porque tu MonocularEstimator espera RGB
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # 1) Generar mapa de profundidad (MDE)
    monocular_depth_val = MonocularEstimator(rgb_frame)

    # 2) Actualizar tracker (le pasamos BGR)
    tracker_result = tracker.update(bgr_frame, draw=False)
    # tracker.update puede devolver either bbox o (bbox, vis) si draw=True
    if tracker_result is not None and len(tracker_result) == 4:
      target_bbox = tracker_result  # (x1, y1, x2, y2)
    else:
      target_bbox = None

    # 3) Encontrar target usando el bbox (si existe)
    target_object_depth_val = FindTargetObject(rgb_frame, target_bbox, monocular_depth_val)

    # 4) Detección de objetos circundantes y cálculo con la referencia
    FindObjects(rgb_frame, target_object_depth_val, monocular_depth_val)

    # 5) Convertir a BGR para escritura (y para que OpenCV muestre colores correctamente)
    out_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    # 6) Anotar info del target en el frame (si está disponible)
    if target_bbox is not None:
        x1, y1, x2, y2 = target_bbox
        cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if tracker.target_id is not None:
            cv2.putText(out_frame, f"ID {tracker.target_id}", (x1, max(0, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Mostrar distancia calculada del target (si FindTargetObject la devolvió)
    if target_object_depth_val is not None:
        try:
            computed = target_object_depth_val[0]  # valor retornado por LensOpticCalculator
            midas_val = target_object_depth_val[1]
            # Ajusta formato y texto según tu unidad real (aquí lo mostramos sin especificar unidad)
            text = f"Target calc: {computed:.2f}"
            text2 = f"MDE val: {midas_val:.3f}"
            # Posicionar el texto sobre la bbox si existe, si no en (10,30)
            if target_bbox is not None:
                tx, ty = x1, max(0, y1 - 30)
            else:
                tx, ty = 10, 30
            cv2.putText(out_frame, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(out_frame, text2, (tx, ty + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        except Exception as e:
            # Si la estructura cambió, evitamos romper el loop
            pass

    # 7) Escribir frame procesado al video de salida
    out.write(out_frame)

    # Progreso simple por consola
    frame_idx += 1
    if frame_idx % 200 == 0:
        print(f"[INFO] Procesados {frame_idx} frames...")

# Liberar recursos
cap.release()
out.release()
end_time = time.time()
elapsed = end_time - start_time
print(f"[INFO] Tiempo total de procesamiento: {elapsed: .2f} segundos")
print(f"[INFO] Procesamiento completado. Video guardado en: {output_path}")
