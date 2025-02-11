import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarnings
warnings.filterwarnings("ignore", category=UserWarning)    # Suppress UserWarnings

import streamlit as st
import numpy as np
import sqlite3
import json
from PIL import Image
import os
import subprocess
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Install system dependencies if not found (for Streamlit Community Cloud)
if not os.path.exists('/usr/bin/tesseract'):
    logger.info("Installing Tesseract-OCR...")
    try:
        subprocess.run(['apt-get', 'update'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(['apt-get', 'install', '-y', 'tesseract-ocr'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing system dependencies: {e}")
        st.error("Failed to install system dependencies. Please check the logs.")
        st.stop()

# Import custom modules
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add project root to Python path
    from models.yolo_detector import YOLODetector
    from ocr.ocr_processor import OCRProcessor
    from utils.config import Config
except ImportError as e:
    logger.error(f"Error importing custom modules: {e}")
    st.error("Failed to load required modules. Please check the logs.")
    st.stop()

# Initialize models
try:
    logger.info("Initializing YOLO model...")
    detector = YOLODetector(Config.model_path)
    logger.info("Initializing OCR processor...")
    ocr_processor = OCRProcessor(language=Config.ocr_languages, psm=Config.ocr_psm)
except Exception as e:
    logger.error(f"Error initializing models: {e}")
    st.error("Failed to initialize models. Please check the logs.")
    st.stop()

# Streamlit app
st.title("Legal Document Digitization")

# File uploader
uploaded_file = st.file_uploader("Upload an Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file:
    try:
        # Open the uploaded file
        logger.info("Processing uploaded file...")
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Run YOLO detection
        logger.info("Running YOLO detection...")
        detections = detector.detect(image)

        # OCR + Error Correction
        logger.info("Running OCR and error correction...")
        extracted_data = ocr_processor.process_detections(image, detections)

        # Display corrected text
        st.subheader("Corrected Extracted Text:")
        st.json(extracted_data)

        # Store in database
        logger.info("Storing results in database...")
        conn = sqlite3.connect(Config.db_path)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bbox TEXT,
                text TEXT
            )
        """)

        # Insert data into the database
        for item in extracted_data:
            if item['text']:  # Only store valid text
                cursor.execute("INSERT INTO ocr_results (bbox, text) VALUES (?, ?)", (str(item['bbox']), item['text']))

        conn.commit()
        conn.close()

        st.success("Corrected text stored in database.")
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error("An error occurred while processing the file. Please check the logs.")
