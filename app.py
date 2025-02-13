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

from utils.config import Config  # Make sure this path is correct
from utils.pdf_processing import process_pdf  # Make sure this path is correct
from utils.image_processing import preprocess_image  # Make sure this path is correct
from models.yolo_detector import YOLODetector  # Make sure this path is correct

# Setup page configuration
st.set_page_config(
    page_title="Legal Document Digitization with YOLO OCR",
    page_icon=":page_facing_up:",
    layout="wide"
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# OCRProcessor class (Now only uses Tesseract)
class OCRProcessor:
    def __init__(self, language='eng', psm=3):
        self.tesseract_config = f'-l {language} --psm {psm}'
        import pytesseract
        self.pytesseract = pytesseract

    def process_detections(self, image, detections):
        results = []
        for detection in detections:
            bbox = detection['bbox']
            roi = self.extract_roi(image, bbox)

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
        return image[int(y):int(y + h), int(x):int(x + w)]


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
    with st.spinner("Loading Tesseract OCR engine..."):
        logger.info("Initializing Tesseract OCR processor")
        return OCRProcessor()  # Initialize Tesseract OCR Processor


def main():
    detector = load_detector()
    ocr_processor = load_ocr_processor()

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")  # Ensure RGB format
            image = np.array(image)
            st.image(image, caption="Uploaded Image")

            detections = detector.detect(image)
            st.write(f"Detections: {detections}")

            if detections:
                ocr_results = ocr_processor.process_detections(image, detections)
                st.write(f"OCR Results: {ocr_results}")

                for result in ocr_results:
                    st.write(f"Text: {result['text']}")
            else:
                st.write("No detections found by YOLO.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.exception(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
