import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import streamlit as st
import numpy as np
import json
from PIL import Image
import os
import subprocess
import logging
import sys
import gc

from utils.config import Config  # Import the Config class
from utils.database_handler import store_ocr_result  # Import the database function

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Install system dependencies (using st.cache_resource)
@st.cache_resource
def install_tesseract():
    if not os.path.exists('/usr/bin/tesseract'):
        logger.info("Installing Tesseract-OCR...")
        try:
            subprocess.run(['apt-get', 'update'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            subprocess.run(['apt-get', 'install', '-y', 'tesseract-ocr'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing Tesseract: {e}")
            return False
    return True

if not install_tesseract():
    st.error("Failed to install Tesseract. Please check the logs.")
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

        for item in extracted_data:
            if item.get('text'):  # Check if 'text' key exists
                corrected_text = item.get('corrected_text')
                success = store_ocr_result(Config.DB_PATH, str(item['bbox']), item['text'], corrected_text)
                if not success:
                    st.error("Error storing OCR result. Check logs.")
            else:
                logger.warning(f"No 'text' found in item: {item}") # Log if 'text' is missing

        st.success("Data stored in database.")

        del image
        gc.collect()

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"An error occurred: {e}")
        st.exception(e)  # Display the full exception details in Streamlit
