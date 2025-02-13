import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import subprocess
import logging
import sys
import gc
import io
import zipfile
import cv2
from paddleocr import PaddleOCR

from utils.config import Config
from utils.pdf_processing import process_pdf
from utils.image_processing import preprocess_image
from models.yolo_detector import YOLODetector

# Setup page configuration
st.set_page_config(
    page_title="Legal Document Digitization with YOLO OCR",
    page_icon=":page_facing_up:",
    layout="wide"
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# Initialize models with improved caching (and placeholder)
@st.cache_resource(max_entries=1)
def load_detector():
    with st.spinner("Loading YOLO model..."):  # Add spinner
        logger.info("Initializing YOLO model...")
        try:
            detector = YOLODetector(Config.model_path)
            logger.info("YOLO model initialized successfully.")
            return detector
        except Exception as e:
            logger.error(f"Error initializing YOLO model: {e}")
            st.error(f"Error loading YOLO model: {e}") # Show error in Streamlit
            raise  # Re-raise the exception to stop execution

@st.cache_resource(max_entries=1)
def load_ocr_processor():
    with st.spinner("Loading OCR engine..."):  # Add spinner
        logger.info("Starting OCR processor initialization...")
        paddle_available, tesseract_available = check_ocr_systems()

        if paddle_available:
            logger.info("Initializing PaddleOCR processor")
            return OCRProcessor(use_paddle=True)
        elif tesseract_available:
            logger.info("Falling back to Tesseract OCR processor")
            return OCRProcessor(use_paddle=False)
        else:
            error_msg = "Neither PaddleOCR nor Tesseract is available. Please install at least one OCR system."
            logger.error(error_msg)
            st.error(error_msg)
            st.stop()  # Stop execution if no OCR engine is available



# ... (Rest of your OCRProcessor class and check_ocr_systems function remain the same) ...

# ... (display_processed_image function remains the same) ...

def main():
    detector = load_detector()  # Load YOLO detector (with caching and spinner)
    ocr_processor = load_ocr_processor()  # Load OCR engine (with caching and spinner)

    # ... (Rest of your main function code remains the same) ...

if __name__ == "__main__":
    main()
