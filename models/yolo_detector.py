import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os

os.environ['ULTRALYTICS_SKIP_SIGNAL_HANDLERS'] = '1'

class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.2, iou_threshold=0.3):
        """
        Initializes the YOLOv8 model.  Handles device selection.
        """
        self.model = YOLO(model_path, task='detect')
        self.model.fuse()  # Optimize model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Determine device (CPU or CUDA) *ONCE* during initialization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)  # Move model to device

    def detect(self, image):
        """
        Runs YOLO detection on the input image. Uses the correct device.
        """
        results = self.model.predict(image, conf=self.conf_threshold, iou=self.iou_threshold, device=self.device)  # Use self.device
        detections = []

        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.cpu().numpy() # Always convert to CPU before numpy
                label = int(cls)
                detections.append({'bbox': [x1, y1, x2, y2], 'class': label, 'confidence': conf})

        return detections
