import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import streamlit as st
import numpy as np
from PIL import Image
import os
import subprocess
import logging
import sys
import gc

from utils.config import Config  # Import the Config class

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

#Check if tesseract and it's files are correctly installed
def check_tesseract_installed():
    try:
        subprocess.run(['tesseract', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

if not check_tesseract_installed():
    st.error("Tesseract-OCR is not installed. Please ensure it is installed via requirements.txt or packages.txt.")
    st.stop()

# Import custom modules (improved path handling)
try:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(module_dir)
    from models.yolo_detector import YOLODetector
    from ocr.ocr_processor import OCRProcessor
except ImportError as e:
    logger.error(f"Error importing custom modules: {e}")
    st.error(f"Failed to load required modules: {e}")
    st.stop()

# Initialize models (using st.cache_resource)
@st.cache_resource
def load_detector():
    logger.info("Initializing YOLO model...")
    try:
        detector = YOLODetector(Config.model_path)
        logger.info("YOLO model initialized successfully.")
        return detector
    except Exception as e:
        logger.error(f"Error initializing YOLO model: {e}")
        st.error(f"Failed to initialize YOLO model: {e}")
        st.stop()

@st.cache_resource
def load_ocr_processor(language, psm):
    logger.info("Starting OCR processor initialization...")
    try:
        ocr_processor = OCRProcessor(language=language, psm=psm)
        logger.info("OCR processor initialized successfully.")
        return ocr_processor
    except Exception as e:
        logger.error(f"Error initializing OCR processor: {e}")
        st.error(f"Failed to initialize OCR processor: {e}")
        st.stop()

# Streamlit app
st.title("Legal Document Digitization")

detector = load_detector()
ocr_processor = load_ocr_processor(Config.ocr_languages, Config.ocr_psm)

uploaded_file = st.file_uploader("Upload an Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image = np.array(image)

        with st.spinner("Processing..."):
            detections = detector.detect(image)
            extracted_data = ocr_processor.process_detections(image, detections)

        st.subheader("Extracted Data:")
        st.json(extracted_data)  # Display extracted data as JSON

        st.subheader("Raw Text:")
        for item in extracted_data:
            if item.get('text'):  # Check if 'text' key exists
                st.text_area("Raw Text", item['text'], height=100)

        st.subheader("Corrected Text:")
        for item in extracted_data:
            if item.get('corrected_text'):  # Check if 'corrected_text' key exists
                st.text_area("Corrected Text", item['corrected_text'], height=100)

        st.subheader("Stamp and Signature Detection:")
        for item in extracted_data:
            if item.get('class') in ['stamp', 'signature']:
                st.write(f"{item['class'].capitalize()}: {'Detected' if item['detected'] else 'Not Detected'}")

        del image
        gc.collect()

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"An error occurred: {e}")
        st.exception(e)  # Display the full exception details in Streamlit
