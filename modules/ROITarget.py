# modules/ROITarget.py
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import cv2

class TargetTracker:
    def __init__(self,
                 yolo_model,   # ahora pasamos la instancia de YOLO
                 max_age=30,
                 conf_thresh=0.35,
                 sim_thresh=0.7,
                 max_missed=60):
        self.yolo = yolo_model
        self.tracker = DeepSort(max_age=max_age)
        self.target_id = None
        self.target_embedding = None
        self.conf_thresh = conf_thresh
        self.sim_thresh = sim_thresh
        self.max_missed = max_missed
        self.missed_frames = 0

    def update(self, frame, draw=False):
        """
        frame: imagen BGR (OpenCV)
        returns: (found_bbox, detections)
            found_bbox: (x1,y1,x2,y2) del target o None
            detections: list de (x1,y1,x2,y2,conf) para todas las personas detectadas en el frame
        """
        # --- 1) Detección con YOLO (intentamos prefiltrar por clase=person)
        # Usamos conf y classes param pero luego comprobamos cls por seguridad
        results = self.yolo(frame, conf=self.conf_thresh, classes=[0], verbose=False)
        persons = []
        for r in results:
            for box in r.boxes:
                # box.xyxy, box.conf, box.cls pueden ser tensores; convertimos
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].tolist())
                cls_id = int(box.cls[0].tolist())
                # doble filtro: param classes+[check]
                if cls_id != 0:
                    continue
                if conf < self.conf_thresh:
                    continue
                persons.append((x1, y1, x2, y2, conf))

        # --- 2) Preparar lista para DeepSort: [([x,y,w,h], conf, "person"), ...]
        ds_detections = []
        for (x1, y1, x2, y2, conf) in persons:
            w = x2 - x1; h = y2 - y1
            ds_detections.append(([x1, y1, w, h], conf, "person"))

        # --- 3) Update DeepSort
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        found_bbox = None
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()
            x1_t, y1_t, x2_t, y2_t = map(int, ltrb)
            emb = track.get_feature()
            if emb is None:
                continue
            norm_emb = emb / (np.linalg.norm(emb) + 1e-6)

            # Asignación / actualización de target
            if self.target_id is None:
                self.target_id = tid
                self.target_embedding = norm_emb
                self.missed_frames = 0
                found_bbox = (x1_t, y1_t, x2_t, y2_t)
                break

            if tid == self.target_id:
                self.target_embedding = norm_emb
                self.missed_frames = 0
                found_bbox = (x1_t, y1_t, x2_t, y2_t)
                break

            sim = float(np.dot(self.target_embedding, norm_emb))
            if sim > self.sim_thresh:
                self.target_id = tid
                self.target_embedding = norm_emb
                self.missed_frames = 0
                found_bbox = (x1_t, y1_t, x2_t, y2_t)
                break

        if found_bbox is None:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed:
                self.target_id = None
                self.target_embedding = None
                self.missed_frames = 0

        # Para que otras partes (FindObjects) no tengan que redetectar:
        # devolvemos las detecciones de persona en coordenadas x1,y1,x2,y2,conf
        return found_bbox, persons
