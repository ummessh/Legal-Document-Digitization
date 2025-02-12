import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

os.environ['ULTRALYTICS_SKIP_SIGNAL_HANDLERS'] = '1'

class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.2, iou_threshold=0.3, input_size=640):
        """Initializes the YOLOv8 model with optimized preprocessing."""
        self.model = YOLO(model_path, task='detect')
        self.model.fuse()  # Optimize model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.class_names = self.model.names

    def preprocess(self, image):
        """Resizes the input image while maintaining aspect ratio."""
        h, w, _ = image.shape
        scale = self.input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized  # Place resized image at the top-left
        return padded, scale

    def detect(self, image):
        """Runs YOLO detection with optimized input processing."""
        start_time = time.time()
        
        preprocessed_image, scale = self.preprocess(image)
        results = self.model.predict(preprocessed_image, conf=self.conf_threshold, iou=self.iou_threshold, device=self.device)
        detections = []

        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1 / scale, y1 / scale, x2 / scale, y2 / scale])  # Scale back to original size
                label = int(cls)
                class_name = self.class_names[label]

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': class_name,
                    'confidence': conf
                })
        
        print(f"YOLO Detection Time: {time.time() - start_time:.2f} sec")
        return detections
