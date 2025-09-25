# modules/ROITarget.py
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import cv2

class TargetTracker:
    def __init__(self,
                 model_path="yolov8n.pt",
                 max_age=30,
                 conf_thresh=0.35,
                 sim_thresh=0.7,
                 max_missed=60):
        """
        model_path: ruta al .pt de ultralytics
        conf_thresh: umbral mínimo de confianza YOLO para pasar a DeepSort
        sim_thresh: umbral de similitud coseno para reasignar target
        max_missed: cuantos frames sin target antes de resetear target_id
        """
        self.yolo = YOLO(model_path)
        self.tracker = DeepSort(max_age=max_age)
        self.target_id = None
        self.target_embedding = None
        self.conf_thresh = conf_thresh
        self.sim_thresh = sim_thresh
        self.max_missed = max_missed
        self.missed_frames = 0

    def update(self, frame, draw=False):
        """
        frame: imagen BGR (como venía del VideoCapture)
        draw: si True devuelve también una copia del frame con el bbox dibujado
        returns:
            if draw: (bbox, vis_frame) where bbox is (x1,y1,x2,y2) or None
            else: bbox or None
        """
        # 1) Inference YOLO (person class only)
        results = self.yolo(frame, classes=[0], verbose=False)
        detections = []
        for r in results[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            conf = float(r.conf[0])
            if conf < self.conf_thresh:
                continue
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        # 2) Update DeepSORT
        tracks = self.tracker.update_tracks(detections, frame=frame)

        found_bbox = None

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            ltrb = track.to_ltrb()  # left, top, right, bottom
            x1, y1, x2, y2 = map(int, ltrb)
            emb = track.get_feature()

            # Normalize embedding
            norm_emb = emb / (np.linalg.norm(emb) + 1e-6)

            # Caso: primer target confirmado -> lo asignamos
            if self.target_id is None:
                self.target_id = tid
                self.target_embedding = norm_emb
                self.missed_frames = 0
                found_bbox = (x1, y1, x2, y2)
                break

            # Si este track es el target actual
            if tid == self.target_id:
                self.target_embedding = norm_emb  # actualizar embedding
                self.missed_frames = 0
                found_bbox = (x1, y1, x2, y2)
                break

            # Si no es el target, comprobar similitud
            sim = float(np.dot(self.target_embedding, norm_emb))
            if sim > self.sim_thresh:
                # reasignamos el target al nuevo tid (es muy parecido)
                self.target_id = tid
                self.target_embedding = norm_emb
                self.missed_frames = 0
                found_bbox = (x1, y1, x2, y2)
                break

        # Si no encontramos el target en este frame
        if found_bbox is None:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed:
                # resetear target si lleva mucho perdido
                self.target_id = None
                self.target_embedding = None
                self.missed_frames = 0

        if draw:
            vis = frame.copy()
            if found_bbox is not None:
                x1, y1, x2, y2 = found_bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if self.target_id is not None:
                    cv2.putText(vis, f"ID {self.target_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return found_bbox, vis

        return found_bbox
