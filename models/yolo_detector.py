import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import pytesseract
from dataclasses import dataclass
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectedRegion:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    text_content: str = ""
    page_number: int = 1

class YOLOTextProcessor:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        logger.info("Initializing YOLOTextProcessor with model: %s", model_path)
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = ['text', 'table', 'stamps', 'signatures']

    def process_image(self, image: np.ndarray, options: Dict, page_num: int = 1) -> List[DetectedRegion]:
        logger.info("Processing image for page number %d", page_num)
        results = self.model(image)[0]
        detected_regions = []
        
        has_stamp = False
        has_signature = False
        
        for box in results.boxes:
            confidence = float(box.conf.item())
            if confidence < self.conf_threshold:
                logger.debug("Skipping box with confidence %.2f", confidence)
                continue
            
            class_id = int(box.cls.item())
            if class_id >= len(self.class_names):
                logger.error("Invalid class ID: %d", class_id)
                continue
            
            class_name = self.class_names[class_id]
            logger.info("Detected %s with confidence %.2f", class_name, confidence)
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            if class_name == 'stamps':
                has_stamp = True
            elif class_name == 'signatures':
                has_signature = True
            elif class_name in ['text', 'table']:
                region_image = image[y1:y2, x1:x2]
                config = f"--psm {options.get('psm', 6)}"
                logger.debug("Extracting text from %s region", class_name)
                try:
                    text = pytesseract.image_to_string(region_image, config=config, lang=options.get('language', 'eng'))
                except Exception as e:
                    logger.error("OCR extraction failed: %s", str(e))
                    text = ""
                
                detected_regions.append(
                    DetectedRegion(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        text_content=text.strip(),
                        page_number=page_num
                    )
                )
        
        logger.info("Stamps detected: %s, Signatures detected: %s", has_stamp, has_signature)
        detected_regions.append(
            DetectedRegion(class_name='stamps', confidence=1.0, bbox=(0,0,0,0), text_content=str(has_stamp), page_number=page_num)
        )
        detected_regions.append(
            DetectedRegion(class_name='signatures', confidence=1.0, bbox=(0,0,0,0), text_content=str(has_signature), page_number=page_num)
        )
        
        return detected_regions

