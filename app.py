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
from utils.ocr_processor import OCRProcessor  
from models.yolo_detector import YOLODetector  # The YOLOv8s model

# Setup page configuration
st.set_page_config(
    page_title="Legal Document Digitization with YOLO OCR",
    page_icon=":page_facing_up:",
    layout="wide"
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Check if Tesseract is installed
def check_tesseract_installed():
    try:
        subprocess.run(['tesseract', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

if not check_tesseract_installed():
    st.error("Tesseract-OCR is not installed. Please ensure it is installed.")
    st.stop()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)  

# Visualization: display original and processed images side by side
def display_processed_image(original_image, processed_image):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)
    with col2:
        st.subheader("Processed Image")
        st.image(processed_image, use_container_width=True)

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
def load_ocr_processor():
    logger.info("Starting OCR processor initialization...")
    try:
        ocr_processor = OCRProcessor(language=Config.ocr_languages, psm=Config.ocr_psm)
        logger.info("OCR processor initialized successfully.")
        return ocr_processor
    except Exception as e:
        logger.error(f"Error initializing OCR processor: {e}")
        st.error(f"Failed to initialize OCR processor: {e}")
        st.stop()

detector = load_detector()
ocr_processor = load_ocr_processor()

st.title("Legal Document Digitization with YOLO OCR")
st.write("By Aryan Tandon and Umesh Tiwari")

# File uploader: allow multiple uploads
uploaded_files = st.file_uploader(
    "Upload Images or PDF files",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True
)

# Containers to hold extracted text outputs for download
all_extracted_texts = []
individual_texts = {}

if uploaded_files:
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"### Processing File: {uploaded_file.name}")
        if uploaded_file.type == "application/pdf":
            with st.spinner("Processing PDF..."):
                try:
                    extracted_text = process_pdf(uploaded_file, Config.ocr_options)
                    st.subheader("Extracted Text from PDF:")
                    st.text_area("Extracted Text", extracted_text, height=300)
                    all_extracted_texts.append(f"File: {uploaded_file.name}\n\n{extracted_text}\n\n{'='*50}\n")
                    individual_texts[uploaded_file.name] = extracted_text
                except Exception as e:
                    logger.error(f"Error processing PDF file {uploaded_file.name}: {e}")
                    st.error(f"An error occurred while processing PDF: {e}")
        else:
            try:
                # Load the original image and create a copy for processing
                original_image = np.array(Image.open(uploaded_file).convert("RGB"))
                processed_image = preprocess_image(
                    original_image.copy(),
                    {
                        'apply_threshold': True,
                        'apply_deskew': True,
                        'apply_denoise': True,
                        'apply_contrast': True
                    }
                )
                
                # Display original vs. processed image
                display_processed_image(original_image, processed_image)
                
                with st.spinner("Processing Image..."):
                    detections = detector.detect(processed_image)
                    extracted_data = ocr_processor.process_detections(processed_image, detections)
                
                # Combine extracted text from detection outputs
                combined_text = ""
                for item in extracted_data:
                    if 'text' in item:
                        combined_text += f"Raw Text (BBox: {item.get('bbox', 'N/A')}):\n{item['text']}\n\n"
                    if 'corrected_text' in item:
                        combined_text += f"Corrected Text (BBox: {item.get('bbox', 'N/A')}):\n{item['corrected_text']}\n\n"
                
                st.subheader("Extracted Data")
                st.text_area("Extracted Text", combined_text, height=300)
                st.subheader("Detailed JSON Output")
                st.json(extracted_data)
                
                all_extracted_texts.append(f"File: {uploaded_file.name}\n\n{combined_text}\n\n{'='*50}\n")
                individual_texts[uploaded_file.name] = combined_text
                
                # Cleanup
                del original_image, processed_image
                gc.collect()
            except Exception as e:
                logger.error(f"Error processing image file {uploaded_file.name}: {e}")
                st.error(f"An error occurred while processing image: {e}")
        progress_bar.progress((i+1)/total_files)
    
    progress_bar.empty()
    
    # Download button for combined extracted text
    combined_text_all = "\n".join(all_extracted_texts)
    combined_text_io = io.BytesIO(combined_text_all.encode('utf-8'))
    st.download_button(
        label="Download Combined Extracted Text",
        data=combined_text_io,
        file_name="combined_extracted_text.txt",
        mime="text/plain"
    )
    
    # Download button for individual extracted texts as a ZIP file
    if individual_texts:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_name, text in individual_texts.items():
                zip_file.writestr(f"{file_name}_extracted.txt", text)
        st.download_button(
            label="Download Individual Extracted Texts",
            data=zip_buffer.getvalue(),
            file_name="individual_extracted_texts.zip",
            mime="application/zip"
        )
