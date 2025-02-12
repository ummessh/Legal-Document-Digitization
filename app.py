import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch  # Import torch explicitly
import streamlit as st
import numpy as np
import sqlite3
import json
from PIL import Image
import os
import subprocess
import logging
import sys
import gc  # For garbage collection

# Set up logging (improved)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout) # Log to stdout for Streamlit
logger = logging.getLogger(__name__)

# Configuration class (using st.cache_data for database path)
class Config:
    model_path = "./models/yolov8n.pt"
    ocr_languages = "eng+hin+mar"
    ocr_psm = 6

    @staticmethod
    @st.cache_data # Cache the db path so it is not recomputed
    def get_db_path():
        return "./data/ocr.db"

    @staticmethod
    def ensure_directories_exist():
        os.makedirs(os.path.dirname(Config.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(Config.get_db_path()), exist_ok=True) # Use cached path

Config.ensure_directories_exist()

# Install system dependencies (using st.cache_resource)
@st.cache_resource  # Cache the installation result
def install_tesseract():
    if not os.path.exists('/usr/bin/tesseract'):
        logger.info("Installing Tesseract-OCR...")
        try:
            subprocess.run(['apt-get', 'update'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            subprocess.run(['apt-get', 'install', '-y', 'tesseract-ocr'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True  # Installation successful
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing system dependencies: {e}")
            return False  # Installation failed
    return True # Already installed

if not install_tesseract():
    st.error("Failed to install system dependencies. Please check the logs.")
    st.stop()



# Import custom modules (improved path handling)
try:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(module_dir) # Make sure the current directory is in the python search path
    from models.yolo_detector import YOLODetector
    from ocr.ocr_processor import OCRProcessor
except ImportError as e:
    logger.error(f"Error importing custom modules: {e}")
    st.error(f"Failed to load required modules: {e}")  # Show the actual error in Streamlit
    st.stop()

# Initialize models (using st.cache_resource, and allowing model selection)
@st.cache_resource # cache the detector
def load_detector(model_path):
    logger.info(f"Initializing YOLO model from {model_path}...")
    try:
        detector = YOLODetector(model_path)
        logger.info("YOLO model initialized successfully.")
        return detector
    except Exception as e:
        logger.error(f"Error initializing YOLO model: {e}")
        st.error(f"Failed to initialize YOLO model: {e}")
        st.stop()

@st.cache_resource # cache the ocr_processor
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

model_choice = st.selectbox("Select YOLO Model", ["yolov8n", "yolov8s", "yolov8m"]) # Allow model selection
model_path = f"./models/{model_choice}.pt" # dynamically build model path

detector = load_detector(model_path)
ocr_processor = load_ocr_processor(Config.ocr_languages, Config.ocr_psm)


uploaded_file = st.file_uploader("Upload an Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB") # Ensure RGB format
        image = np.array(image)

        with st.spinner("Processing..."):  # Show a spinner
            detections = detector.detect(image)
            extracted_data = ocr_processor.process_detections(image, detections)

        st.subheader("Corrected Extracted Text:")
        st.json(extracted_data)

        # Store in database (using a context manager for the connection)
        db_path = Config.get_db_path() # Get the (cached) path
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ocr_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bbox TEXT,
                    text TEXT
                )
            """)
            for item in extracted_data:
                if item.get('text'):  # Safer way to check for 'text' key
                    cursor.execute("INSERT INTO ocr_results (bbox, text) VALUES (?, ?)", (str(item['bbox']), item['text']))
            conn.commit()

        st.success("Corrected text stored in database.")

        del image # delete image from memory
        gc.collect() # run garbage collection


    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"An error occurred: {e}") # Show specific error in Streamlit
        st.exception(e) # print the traceback in streamlit
