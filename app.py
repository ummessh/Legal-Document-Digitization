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

# Check if tesseract is installed (improved)
def check_tesseract_installed():
    try:
        subprocess.run(['tesseract', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

if not check_tesseract_installed():
    st.error("Tesseract-OCR is not installed. Please ensure it is installed.")  # More concise message
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
    # ... (same as before)

@st.cache_resource
def load_ocr_processor(language, psm):
    # ... (same as before)

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
            detections = detector.detect(image)  # Get all detections
            extracted_data = ocr_processor.process_detections(image, detections)

            # --- Displaying extracted data ---
            st.subheader("Extracted Data:")
            st.json(extracted_data)  # Display all data as JSON (for debugging)

            # --- More user-friendly display ---
            for item in extracted_data:
                if 'text' in item:  # Check if text is present
                    st.subheader("Raw Text:")
                    st.text_area(f"Raw Text (Bounding Box: {item.get('bbox', 'N/A')})", item['text'], height=100)  # Include bbox info

                if 'corrected_text' in item:  # Check if corrected text is present
                    st.subheader("Corrected Text:")
                    st.text_area(f"Corrected Text (Bounding Box: {item.get('bbox', 'N/A')})", item['corrected_text'], height=100)  # Include bbox info

                # Handle stamps, signatures, and tables:
                if 'class' in item: # Check if class is present
                    if item['class'] in ['stamp', 'signature']:
                        st.subheader(f"{item['class'].capitalize()} Detection:")
                        st.write(f"{item['class'].capitalize()}: {'Detected' if item.get('detected', False) else 'Not Detected'}") # More robust check for 'detected'

                    elif item['class'] == 'table':
                        st.subheader("Table Data:")
                        if 'cells' in item:
                            st.write(item['cells']) # Or format the table data better
                        else:
                            st.write("No cell data found.")

        del image
        gc.collect()

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"An error occurred: {e}")
        st.exception(e)  # Display full exception details
