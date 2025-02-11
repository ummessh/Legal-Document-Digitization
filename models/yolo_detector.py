import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os

os.environ['ULTRALYTICS_SKIP_SIGNAL_HANDLERS'] = '1'

class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.2, iou_threshold=0.3):
        """
        Initializes the YOLOv8 model with TensorRT acceleration.
        """
        self.model = YOLO(model_path,task='detect')
        self.model.fuse()  # Optimize model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image):
        """
        Runs YOLO detection on the input image.
        """
        results = self.model.predict(image, conf=self.conf_threshold, iou=self.iou_threshold, device='cuda')
        detections = []
        
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                label = int(cls)
                detections.append({'bbox': [x1, y1, x2, y2], 'class': label, 'confidence': conf})
        
        return detections
