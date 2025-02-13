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

# Define check_ocr_systems FIRST (before it's used)
def check_ocr_systems():
    paddle_available = True
    tesseract_available = True

    try:
        _ = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)  # Suppress PaddleOCR logs
    except Exception:
        paddle_available = False

    try:
        subprocess.run(['tesseract', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        tesseract_available = False

    return paddle_available, tesseract_available


# OCRProcessor class
class OCRProcessor:
    def __init__(self, language='eng', psm=3, use_paddle=True):
        self.use_paddle = use_paddle
        if use_paddle:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)  # Suppress PaddleOCR logs
                logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing PaddleOCR: {e}")
                self.use_paddle = False

        if not self.use_paddle:
            self.tesseract_config = f'-l {language} --psm {psm}'
            import pytesseract
            self.pytesseract = pytesseract

    def process_detections(self, image, detections):
        results = []
        for detection in detections:
            bbox = detection['bbox']
            roi = self.extract_roi(image, bbox)

            if self.use_paddle:
                try:
                    paddle_result = self.paddle_ocr.ocr(roi, cls=True)
                    if paddle_result and paddle_result[0]:
                        text = '\n'.join([line[1][0] for line in paddle_result[0]])
                    else:
                        text = ''
                except Exception as e:
                    logger.error(f"PaddleOCR processing error: {e}")
                    text = ''
            else:
                try:
                    text = self.pytesseract.image_to_string(roi, config=self.tesseract_config)
                except Exception as e:
                    logger.error(f"Tesseract processing error: {e}")
                    text = ''

            results.append({
                'bbox': bbox,
                'text': text,
                'corrected_text': text  # Add your text correction logic here if needed
            })
        return results

    @staticmethod
    def extract_roi(image, bbox):
        x, y, w, h = bbox
        return image[int(y):int(y+h), int(x):int(x+w)]


# Initialize models with improved caching
@st.cache_resource(max_entries=1)
def load_detector():
    with st.spinner("Loading YOLO model..."):
        logger.info("Initializing YOLO model...")
        try:
            detector = YOLODetector(Config.model_path)
            logger.info("YOLO model initialized successfully.")
            return detector
        except Exception as e:
            logger.error(f"Error initializing YOLO model: {e}")
            st.error(f"Error loading YOLO model: {e}")
            raise  # Re-raise the exception to stop execution

@st.cache_resource(max_entries=1)
def load_ocr_processor():
    with st.spinner("Loading OCR engine..."):
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


def main():
    detector = load_detector()  # Load YOLO detector (with caching and spinner)
    ocr_processor = load_ocr_processor()  # Load OCR engine (with caching and spinner)

if __name__ == "__main__":
    main()
