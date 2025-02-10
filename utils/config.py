import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "../models")
    OCR_DIR = os.path.join(BASE_DIR, "../ocr")
    DB_PATH = os.path.join(BASE_DIR, "../database/ocr_results.db")
    
    # YOLO Model
    MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    
    # OCR Settings
    OCR_LANGUAGES = "eng+hin+mar"
    OCR_PSM = 6  # Default Page Segmentation Mode
    
    # mT5-based Text Correction
    USE_MT5_CORRECTION = True  # Enable or disable mT5 correction
    MT5_MODEL_PATH = os.path.join(BASE_DIR, "../models/mt5-correction")  # Path to fine-tuned mT5 model
    
    # Database
    TABLE_CREATION_QUERY = """
    CREATE TABLE IF NOT EXISTS ocr_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bbox TEXT,
        text TEXT
    );
    """
