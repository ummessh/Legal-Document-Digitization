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
from paddleocr import PaddleOCR

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

# Modified OCR Processor class with PaddleOCR integration
class OCRProcessor:
    def __init__(self, language='eng', psm=3, use_paddle=True):
        self.use_paddle = use_paddle
        if use_paddle:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
                logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing PaddleOCR: {e}")
                self.use_paddle = False
                
        if not self.use_paddle:
            self.tesseract_config = f'-l {language} --psm {psm}'
            import pytesseract
            self.pytesseract = pytesseract

    def process_detections(self, image, detections):
        results = []
        for detection in detections:
            bbox = detection['bbox']
            roi = self.extract_roi(image, bbox)
            
            if self.use_paddle:
                try:
                    paddle_result = self.paddle_ocr.ocr(roi, cls=True)
                    if paddle_result and paddle_result[0]:
                        text = '\n'.join([line[1][0] for line in paddle_result[0]])
                    else:
                        text = ''
                except Exception as e:
                    logger.error(f"PaddleOCR processing error: {e}")
                    text = ''
            else:
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
        return image[int(y):int(y+h), int(x):int(x+w)]

# Check OCR system availability
def check_ocr_systems():
    paddle_available = True
    tesseract_available = True
    
    try:
        _ = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    except Exception:
        paddle_available = False
        
    try:
        subprocess.run(['tesseract', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        tesseract_available = False
    
    return paddle_available, tesseract_available

# Initialize models with improved caching
@st.cache_resource(max_entries=1)
def load_detector():
    logger.info("Initializing YOLO model...")
    try:
        detector = YOLODetector(Config.model_path)
        logger.info("YOLO model initialized successfully.")
        return detector
    except Exception as e:
        logger.error(f"Error initializing YOLO model: {e}")
        raise e

@st.cache_resource(max_entries=1)
def load_ocr_processor():
    logger.info("Starting OCR processor initialization...")
    paddle_available, tesseract_available = check_ocr_systems()
    
    if paddle_available:
        logger.info("Initializing PaddleOCR processor")
        return OCRProcessor(use_paddle=True)
    elif tesseract_available:
        logger.info("Falling back to Tesseract OCR processor")
        return OCRProcessor(use_paddle=False)
    else:
        error_msg = "Neither PaddleOCR nor Tesseract is available. Please install at least one OCR system."
        logger.error(error_msg)
        st.error(error_msg)
        st.stop()

def display_processed_image(original_image, processed_image):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)
    with col2:
        st.subheader("Processed Image")
        st.image(processed_image, use_container_width=True)

def main():
    detector = load_detector()
    ocr_processor = load_ocr_processor()

    st.title("Legal Document Digitization with YOLO OCR")
    st.write("By Aryan Tandon and Umesh Tiwari")

    # OCR System Status
    paddle_available, tesseract_available = check_ocr_systems()
    st.sidebar.subheader("OCR System Status")
    st.sidebar.write(f"PaddleOCR: {'✅ Available' if paddle_available else '❌ Not Available'}")
    st.sidebar.write(f"Tesseract: {'✅ Available' if tesseract_available else '❌ Not Available'}")

    uploaded_files = st.file_uploader(
        "Upload Images or PDF files",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True
    )

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

                    display_processed_image(original_image, processed_image)

                    with st.spinner("Processing Image..."):
                        detections = detector.detect(processed_image)
                        extracted_data = ocr_processor.process_detections(processed_image, detections)

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

                    del original_image, processed_image
                    gc.collect()
                except Exception as e:
                    logger.error(f"Error processing image file {uploaded_file.name}: {e}")
                    st.error(f"An error occurred while processing image: {e}")
            
            progress_bar.progress((i+1)/total_files)

        progress_bar.empty()

        combined_text_all = "\n".join(all_extracted_texts)
        combined_text_io = io.BytesIO(combined_text_all.encode('utf-8'))
        st.download_button(
            label="Download Combined Extracted Text",
            data=combined_text_io,
            file_name="combined_extracted_text.txt",
            mime="text/plain"
        )

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

if __name__ == "__main__":
    main()
