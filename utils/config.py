import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "../models")
    OCR_DIR = os.path.join(BASE_DIR, "../ocr")

    # YOLO Model
    model_path = os.path.join(MODEL_DIR, "best.pt")
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4

    # OCR Settings
    ocr_languages = "eng+hin+mar"  # Limit OCR to these languages
    ocr_psm = 6  # Default Page Segmentation Mode
