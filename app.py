import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import logging
import sys
import io
import cv2
import pandas as pd
import fitz
import camelot

from utils.config import Config
from utils.pdf_processing import process_pdf
from utils.image_processing import preprocess_image
from models.yolo_detector import YOLODetector

# Page configuration
st.set_page_config(
    page_title="Legal Document Digitization with YOLO OCR",
    page_icon=":page_facing_up:",
    layout="wide"
)

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def get_supported_languages():
    """Returns a dictionary of supported languages and their codes."""
    return {
        'English': 'eng',
        'Hindi': 'hin',
        'Marathi': 'mar'
    }

class OCRProcessor:
    def __init__(self, language='eng', psm=3):
        self.tesseract_config = f'-l {language} --psm {psm}'
        try:
            import pytesseract
            self.pytesseract = pytesseract
        except ImportError:
            logger.error("pytesseract is not installed. Please install it using 'pip install pytesseract'")
            st.error("pytesseract is not installed. Please install it.")
            sys.exit(1)

    def update_config(self, language, psm):
        """Update Tesseract configuration with new language and PSM."""
        self.tesseract_config = f'-l {language} --psm {psm}'

    def process_detections(self, image, detections, preprocessing_options=None):
        results = []
        for detection in detections:
            bbox = detection['bbox']
            roi = self.extract_roi(image, bbox)
            preprocessed_roi = preprocess_image(roi, preprocessing_options)

            try:
                text = self.pytesseract.image_to_string(preprocessed_roi, config=self.tesseract_config)
                results.append({
                    'bbox': bbox,
                    'text': text.strip(),
                    'corrected_text': text.strip()  # Placeholder for later correction
                })
            except Exception as e:
                logger.error(f"Tesseract processing error: {e}")
                results.append({
                    'bbox': bbox,
                    'text': '',
                    'corrected_text': ''
                })

        return results

    @staticmethod
    def extract_roi(image, bbox):
        x, y, w, h = bbox
        return image[int(y):int(y + h), int(x):int(x + w)]

def extract_table(image):
    """Extract tables from image using appropriate table extraction method."""
    try:
        # Save image temporarily
        temp_path = "temp_table.png"
        cv2.imwrite(temp_path, image)
        
        # Use camelot for table extraction
        tables = camelot.read_pdf(temp_path, flavor='stream')
        
        if len(tables) > 0:
            df = tables[0].df
        else:
            df = pd.DataFrame()
            
        # Clean up
        os.remove(temp_path)
        return df
    except Exception as e:
        logger.error(f"Table extraction failed: {e}")
        return pd.DataFrame()

@st.cache_resource(max_entries=1)
def load_detector():
    with st.spinner("Loading YOLO model..."):
        logger.info("Initializing YOLO model...")
        try:
            detector = YOLODetector(Config.model_path)
            logger.info("YOLO model initialized successfully.")
            return detector
        except Exception as e:
            logger.error(f"Error initializing YOLO model: {e}")
            st.error(f"Error loading YOLO model: {e}")
            raise

@st.cache_resource(max_entries=1)
def load_ocr_processor():
    with st.spinner("Loading Tesseract OCR engine..."):
        logger.info("Initializing Tesseract OCR processor")
        return OCRProcessor()

def display_confidence_scores(detections):
    """Display confidence scores for different detection types."""
    st.write("## Confidence Scores:")
    confidence_dict = {}
    for detection in detections:
        if 'class' in detection:
            confidence_dict[detection['class']] = detection['confidence']

    for idx, (class_name, default) in enumerate([
        ('text', 'null'),
        ('table', 'null'),
        ('stamp', 'null'),
        ('signature', 'null')
    ], 1):
        st.write(f"{idx}) {class_name.title()}: {confidence_dict.get(class_name, default)}")

def process_image(image, detections, ocr_processor, preprocessing_options=None):
    """Process image and return detected elements with their processed results."""
    image_with_boxes = image.copy()
    text_images = []
    table_images = []
    stamp_images = []
    signature_images = []
    text_results = []

    if detections:
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = map(int, bbox)
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

            category = detection.get('class', detection.get('category', 'unknown'))
            cv2.putText(image_with_boxes, str(category), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            roi = image[y:y+h, x:x+w]

            if category == "text":
                ocr_results = ocr_processor.process_detections(image, [detection], preprocessing_options)
                text_results.extend(ocr_results)
                text_images.append(roi)
            elif category == "table":
                df = extract_table(roi)
                if not df.empty:
                    st.dataframe(df)
                table_images.append(roi)
            elif category == "stamp":
                stamp_images.append(roi)
            elif category == "signature":
                signature_images.append(roi)

    return image_with_boxes, text_images, table_images, stamp_images, signature_images, text_results

def display_extracted_entities(text_images, table_images, stamp_images, signature_images):
    """Display all extracted entities with proper formatting."""
    entity_counter = 1
    
    for entity_type, images in [
        ("Text", text_images),
        ("Tables", table_images),
        ("Stamps", stamp_images),
        ("Signatures", signature_images)
    ]:
        if images:
            st.write(f"{entity_type}:")
            for img in images:
                st.write(f"{entity_counter})")
                st.image(img, width=400)
                entity_counter += 1
        else:
            st.write(f"{entity_counter}) {entity_type}: Not Detected")
            entity_counter += 1

def process_pdf_page(page, detector, ocr_processor, preprocessing_options):
    """Process a single PDF page and return results."""
    pix = page.get_pixmap()
    image = Image.open(io.BytesIO(pix.tobytes())).convert("RGB")
    image = np.array(image)
    
    detections = detector.detect(image)
    return process_image(image, detections, ocr_processor, preprocessing_options)

def main():
    detector = load_detector()
    ocr_processor = load_ocr_processor()
    
    st.title("Legal Document Digitizer")
    st.write("By Aryan Tandon and Umesh Tiwari")

    # Inject CSS
    st.markdown("""
        <style>
        .rounded-box {
            border-radius: 10px;
            background-color: #f0f0f0;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        }
        .confidence-box {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .confidence-item {
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
        }
        .entity-image {
            border: 1px solid #ccc;
            margin: 5px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar options
    st.sidebar.title("Document Processing Options")

    # Language settings
    st.sidebar.subheader("Language Settings")
    available_languages = get_supported_languages()
    default_lang = 'English'

    primary_lang = st.sidebar.selectbox(
        "Primary Language",
        options=list(available_languages.keys()),
        index=list(available_languages.keys()).index(default_lang)
    )

    additional_langs = st.sidebar.multiselect(
        "Additional Languages (Optional)",
        options=[lang for lang in available_languages.keys() if lang != primary_lang]
    )

    selected_langs = [primary_lang] + additional_langs
    lang_codes = '+'.join([available_languages[lang] for lang in selected_langs])

    psm = st.sidebar.selectbox(
        "Text Layout Detection",
        options=[3, 4, 6, 11, 12],
        index=0,
        format_func=lambda x: {
            3: "Automatic Detection",
            4: "Single Column Layout",
            6: "Single Text Block",
            11: "Line by Line",
            12: "Word by Word"
        }[x]
    )

    ocr_processor.update_config(lang_codes, psm)

    # Image enhancement options
    st.sidebar.subheader("Image Enhancement Options")
    preprocessing_options = {
        'apply_threshold': st.sidebar.checkbox("Sharpen Text", value=True),
        'apply_deskew': st.sidebar.checkbox("Straighten Document", value=True),
        'apply_denoise': st.sidebar.checkbox("Remove Background Noise", value=True),
        'apply_contrast': st.sidebar.checkbox("Enhance Text Visibility", value=False)
    }

    uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "png", "jpeg", "pdf"])

    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                doc = fitz.open(uploaded_file)
                
                # Initialize session state for pagination
                if 'current_page' not in st.session_state:
                    st.session_state.current_page = 0

                # Page navigation
                if doc.page_count > 1:
                    cols = st.columns(2)
                    with cols[0]:
                        if st.session_state.current_page > 0:
                            if st.button('Previous Page'):
                                st.session_state.current_page -= 1
                    with cols[1]:
                        if st.session_state.current_page < doc.page_count - 1:
                            if st.button('Next Page'):
                                st.session_state.current_page += 1

                # Process current page
                page = doc[st.session_state.current_page]
                progress_text = f"Processing page {st.session_state.current_page + 1}/{doc.page_count}"
                
                with st.spinner(progress_text):
                    results = process_pdf_page(page, detector, ocr_processor, preprocessing_options)
                    image_with_boxes, text_images, table_images, stamp_images, signature_images, text_results = results
                    
                    st.image(image_with_boxes, caption=f"Page {st.session_state.current_page + 1} with Detections", width=400)
                    display_confidence_scores(detector.last_detections)
                    display_extracted_entities(text_images, table_images, stamp_images, signature_images)
                    
                    if text_results:
                        st.write("## Extracted Text:")
                        for result in text_results:
                            st.write(f"Text: {result['text']}")
                    else:
                        st.write("No text detected on this page.")

                doc.close()

            else:  # Image processing
                image = Image.open(uploaded_file).convert("RGB")
                image = np.array(image)
                
                with st.spinner("Processing image..."):
                    detections = detector.detect(image)
                    results = process_image(image, detections, ocr_processor, preprocessing_options)
                    image_with_boxes, text_images, table_images, stamp_images, signature_images, text_results = results
                    
                    st.image(image_with_boxes, caption="Processed Image with Detections", width=400)
                    display_confidence_scores(detections)
                    display_extracted_entities(text_images, table_images, stamp_images, signature_images)
                    
                    if text_results:
                        st.write("## Extracted Text:")
                        for result in text_results:
                            st.write(f"Text: {result['text']}")
                    else:
                        st.write("No text detected in the image.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.exception(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
