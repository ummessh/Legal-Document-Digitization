import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os

os.environ['ULTRALYTICS_SKIP_SIGNAL_HANDLERS'] = '1'

class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.2, iou_threshold=0.3):
        """Initializes the YOLOv8 model."""
        self.model = YOLO(model_path, task='detect')
        self.model.fuse()  # Optimize model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Class names (add these)
        self.class_names = self.model.names

    def detect(self, image):
        """Runs YOLO detection and adds class names."""
        results = self.model.predict(image, conf=self.conf_threshold, iou=self.iou_threshold, device=self.device)
        detections = []

        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]) # Convert to int
                label = int(cls)
                class_name = self.class_names[label]  # Get class name

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': class_name,  # Store class name
                    'confidence': conf
                })

        return detections

    def detect_stamps(self, image):
        """Detects stamps."""
        detections = self.detect(image)
        return [det for det in detections if det['class'] == 'stamp']  # Filter for stamps

    def detect_signatures(self, image):
        """Detects signatures."""
        detections = self.detect(image)
        return [det for det in detections if det['class'] == 'signature']  # Filter for signatures

    def detect_tables(self, image):
        """Detects tables and extracts cell data (placeholder)."""
        detections = self.detect(image)
        table_detections = [det for det in detections if det['class'] == 'table']

        for table_det in table_detections:
            # Placeholder for cell detection and extraction
            # Replace this with your actual logic to find cells within the table
            table_det['cells'] = [] # Replace with your logic to get cell bounding boxes and text
            x1, y1, x2, y2 = map(int, table_det['bbox'])
            table_image = image[y1:y2, x1:x2] # Crop table image

            # Example: Dummy cell data (replace with your actual cell detection)
            # Replace this with your actual logic to get cell bounding boxes and text
            # cell_bounding_boxes = [[10, 10, 50, 30], [60, 10, 100, 30]] # Replace with your cell detection logic
            # for cell_bbox in cell_bounding_boxes:
            #     cx1, cy1, cx2, cy2 = map(int, cell_bbox)
            #     cell_image = table_image[cy1:cy2, cx1:cx2]
            #     # ... (use OCR to extract text from cell_image) ...
            #     table_det['cells'].append({'bbox': cell_bbox, 'text': 'dummy cell text'})

        return table_detections
