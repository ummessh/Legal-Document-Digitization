import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarnings
warnings.filterwarnings("ignore", category=UserWarning)    # Suppress UserWarnings

import torch
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

# Configuration class
class Config:
    # Path to YOLO model (use a smaller model for deployment)
    model_path = "./models/yolov8n.pt"  # Updated to YOLOv8

    # OCR settings
    ocr_languages = "eng+hin+mar"  # Languages for OCR
    ocr_psm = 6                    # Page segmentation mode

    # Path to SQLite database
    db_path = "./data/ocr.db"

    # Ensure the required directories exist
    @staticmethod
    def ensure_directories_exist():
        os.makedirs(os.path.dirname(Config.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(Config.db_path), exist_ok=True)

# Ensure directories exist
Config.ensure_directories_exist()

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
except ImportError as e:
    logger.error(f"Error importing custom modules: {e}")
    st.error("Failed to load required modules. Please check the logs.")
    st.stop()

# Initialize models
try:
    logger.info("Initializing YOLO model...")
    detector = YOLODetector(Config.model_path)
    logger.info("YOLO model initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing YOLO model: {e}")
    st.error("Failed to initialize YOLO model. Please check the logs.")
    st.stop()

try:
    logger.info("Starting OCR processor initialization...")
    ocr_processor = OCRProcessor(language=Config.ocr_languages, psm=Config.ocr_psm)
    logger.info("OCR processor initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing OCR processor: {e}")
    st.error("Failed to initialize OCR processor. Please check the logs.")
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
