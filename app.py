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
from utils.pdf_processing import process_pdf  # Import PDF processing function

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Check if tesseract is installed
def check_tesseract_installed():
    try:
        subprocess.run(['tesseract', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

if not check_tesseract_installed():
    st.error("Tesseract-OCR is not installed. Please ensure it is installed.")
    st.stop()

# --- KEY CHANGE: Add project root to Python path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)  # Insert at the beginning

# Import custom modules
try:
    from models.yolo_detector import YOLODetector
    from ocr.ocr_processor import OCRProcessor
except ImportError as e:
    logger.error(f"Error importing custom modules: {e}")
    st.error(f"Failed to load required modules: {e}")
    st.stop()

# Initialize models
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
        if uploaded_file.type == "application/pdf":
            with st.spinner("Processing PDF..."):
                extracted_text = process_pdf(uploaded_file, Config.ocr_options)
                st.subheader("Extracted Text from PDF:")
                st.text_area("Extracted Text", extracted_text, height=300)
        else:
            image = Image.open(uploaded_file).convert("RGB")
            image = np.array(image)

            with st.spinner("Processing Image..."):
                detections = detector.detect(image)
                extracted_data = ocr_processor.process_detections(image, detections)

                st.subheader("Extracted Data (JSON):")
                st.json(extracted_data)  # Display all data as JSON (for debugging)

                for item in extracted_data:
                    if 'text' in item:
                        st.subheader(f"Raw Text (Bounding Box: {item.get('bbox', 'N/A')})")
                        st.text_area(f"Raw Text", item['text'], height=100)

                    if 'corrected_text' in item:
                        st.subheader(f"Corrected Text (Bounding Box: {item.get('bbox', 'N/A')})")
                        st.text_area(f"Corrected Text", item['corrected_text'], height=100)

                    if 'class' in item:
                        if item['class'] in ['stamp', 'signature']:
                            st.subheader(f"{item['class'].capitalize()} Detection:")
                            st.write(f"{item['class'].capitalize()}: {'Detected' if item.get('detected', False) else 'Not Detected'}")

                        elif item['class'] == 'table':
                            st.subheader(f"Table Data (Bounding Box: {item.get('bbox', 'N/A')})")
                            if 'cells' in item and item['cells']:
                                try:
                                    import pandas as pd
                                    df = pd.DataFrame(item['cells'])
                                    st.dataframe(df)
                                except Exception as e:
                                    st.write("Error displaying table data. Raw data:")
                                    st.write(item['cells'])
                            else:
                                st.write("No cell data found.")

        del image
        gc.collect()

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"An error occurred: {e}")
        st.exception(e)  # Display full exception details
