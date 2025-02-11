import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "../models")
    OCR_DIR = os.path.join(BASE_DIR, "../ocr")
    DB_PATH = os.path.join(BASE_DIR, "../database/ocr_results.db")

    # YOLO Model
    model_path = os.path.join(MODEL_DIR, "best.pt")
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4

    # OCR Settings
    OCR_LANGUAGES = "eng+hin+mar"  # Limit OCR to these languages
    OCR_PSM = 6  # Default Page Segmentation Mode

    # IndicBART-based Text Correction
    USE_INDICBART_CORRECTION = True  # Enable or disable IndicBART correction
    INDICBART_MODEL = "ai4bharat/IndicBART"  # IndicBART model from Hugging Face

    # Database
    TABLE_CREATION_QUERY = """
    CREATE TABLE IF NOT EXISTS ocr_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bbox TEXT,
        text TEXT
    );
    """
