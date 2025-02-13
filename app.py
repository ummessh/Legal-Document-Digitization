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


# OCRProcessor class (Now only uses Tesseract)
class OCRProcessor:
    # ... (No changes to OCRProcessor class) ...

# Initialize models with improved caching
@st.cache_resource(max_entries=1)
def load_detector():
    # ... (No changes to load_detector function) ...

@st.cache_resource(max_entries=1)
def load_ocr_processor():
    # ... (No changes to load_ocr_processor function) ...


def main():
    detector = load_detector()
    ocr_processor = load_ocr_processor()

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        try:  # Add a try-except block for error handling
            image = Image.open(uploaded_image).convert("RGB") # Ensure RGB format
            image = np.array(image)
            st.image(image, caption="Uploaded Image") # Display uploaded image

            detections = detector.detect(image)
            st.write(f"Detections: {detections}")  # Check YOLO output

            if detections:
                ocr_results = ocr_processor.process_detections(image, detections)
                st.write(f"OCR Results: {ocr_results}")  # Check Tesseract output

                for result in ocr_results:
                    st.write(f"Text: {result['text']}")  # Display the extracted text
            else:
                st.write("No detections found by YOLO.")

        except Exception as e: # Handle any exceptions during processing
            st.error(f"An error occurred: {e}") # Display error to user
            logger.exception(f"An error occurred: {e}") # Log the full traceback for debugging


if __name__ == "__main__":
    main()
