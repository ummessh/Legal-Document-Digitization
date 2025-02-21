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

from utils.config import Config 
from utils.pdf_processing import PDFProcessor
from utils.image_processing import preprocess_image
from models.yolo_detector import YOLODetector

st.set_page_config(
    page_title="Legal Document Digitization with YOLO OCR",
    page_icon=":page_facing_up:",
    layout="wide"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def get_supported_languages():
    """Returns a dictionary of supported languages and their codes."""
    return {'English': 'eng',
            'Hindi': 'hin',
            'Marathi':'mar'
           }

class OCRProcessor:
    def __init__(self, language='eng', psm=3):
        self.tesseract_config = f'-l {language} --psm {psm}'
        import pytesseract
        self.pytesseract = pytesseract
        
    def update_config(self, language, psm):
        """Update Tesseract configuration with new language and PSM."""
        self.tesseract_config = f'-l {language} --psm {psm}'

    def process_detections(self, image, detections, preprocessing_options=None):
        results = []
        for detection in detections:
            bbox = detection['bbox']
            roi = self.extract_roi(image, bbox)

            # Preprocess the ROI
            preprocessed_roi = preprocess_image(roi, preprocessing_options)

            try:
                text = self.pytesseract.image_to_string(preprocessed_roi, config=self.tesseract_config)
            except Exception as e:
                logger.error(f"Tesseract processing error: {e}")
                text = ''

            results.append({
                'bbox': bbox,
                'text': text,
                'corrected_text': text
            })
        return results

    @staticmethod
    def extract_roi(image, bbox):
        x, y, w, h = bbox
        return image[int(y):int(y + h), int(x):int(x + w)]
        
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

def process_image(image, detections, ocr_processor, page_num=None, preprocessing_options=None):
    image_with_boxes = image.copy()
    text_images = []
    table_images = []
    stamp_images = []
    signature_images = []

    if detections:
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = map(int, bbox)
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if 'class' in detection:
                category = detection['class']
            elif 'confidence' in detection:
                confidence = detection['confidence']
                if confidence > 0.8:
                    category = "text"
                else:
                    category = "unknown"
            else:
                category = "unknown"

            cv2.putText(image_with_boxes, str(category), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            roi = image[y:y+h, x:x+w]

            if category == "text":
                ocr_results = ocr_processor.process_detections(image, [detection], preprocessing_options)
                for result in ocr_results:
                    st.write(f"Category: {category}, Text: {result['text']}")
                text_images.append(roi)
            elif category == "table":
                try:
                    df = pd.DataFrame()  # Placeholder - Replace with actual conversion
                    st.dataframe(df)
                    table_images.append(roi)
                except Exception as e:
                    st.error(f"Error processing table: {e}")
                    logger.exception(f"Error processing table: {e}")
            elif category == "stamp":
                st.write("Stamp detected!")
                stamp_images.append(roi)
            elif category == "signature":
                st.write("Signature detected!")
                signature_images.append(roi)
            else:
                st.write(f"Category: {category} (Unknown)")
    return image_with_boxes, text_images, table_images, stamp_images, signature_images
    
def main():
    detector = load_detector()
    ocr_processor = load_ocr_processor()
    pdf_processor = PDFProcessor()  # Initialize PDF processor
    
    st.title("Legal Document Digitizer")
    st.write("By Aryan Tandon and Umesh Tiwari")

    # Sidebar for options
    st.sidebar.title("Document Processing Options")

    # Language Selection
    st.sidebar.subheader("Language Settings")
    available_languages = get_supported_languages()
    default_lang = "English"

    # Primary language selection
    primary_lang = st.sidebar.selectbox(
        "Primary Language",
        options=list(available_languages.keys()),
        index=list(available_languages.keys()).index(default_lang),
        help="Select the main language of your document",
    )

    # Additional languages selection
    additional_langs = st.sidebar.multiselect(
        "Additional Languages (Optional)",
        options=[lang for lang in available_languages.keys() if lang != primary_lang],
        help="Select additional languages if your document contains multiple languages",
    )

    # Combine selected languages for Tesseract
    selected_langs = [primary_lang] + additional_langs
    lang_codes = "+".join([available_languages[lang] for lang in selected_langs])

    # PSM Selection
    psm = st.sidebar.selectbox(
        "Text Layout Detection",
        options=[3, 4, 6, 11, 12],
        index=0,
        format_func=lambda x: {
            3: "Automatic Detection",
            4: "Single Column Layout",
            6: "Single Text Block",
            11: "Line by Line",
            12: "Word by Word",
        }[x],
        help="Choose how the system should read your document's layout",
    )

    # Update OCR processor with selected language and PSM
    ocr_processor.update_config(lang_codes, psm)

    # Preprocessing options with better labels
    st.sidebar.subheader("Image Enhancement Options")
    preprocessing_options = {
        "apply_threshold": st.sidebar.checkbox("Sharpen Text", value=True, 
                                            help="Improves text clarity by increasing contrast"),
        "apply_deskew": st.sidebar.checkbox("Straighten Document", value=True, 
                                         help="Corrects tilted or skewed documents"),
        "apply_denoise": st.sidebar.checkbox("Remove Background Noise", value=True,
                                          help="Removes specks and background interference"),
        "apply_contrast": st.sidebar.checkbox("Enhance Text Visibility", value=False,
                                           help="Boosts text brightness and contrast")
    }

    uploaded_file = st.file_uploader(
        "Choose an image or PDF...", type=["jpg", "png", "jpeg", "pdf"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                # Process PDF using PDFProcessor
                for page_num, image_np, text in pdf_processor.process_pdf(uploaded_file, preprocessing_options):
                    if image_np is not None:
                        # Display the processed page
                        st.image(image_np, caption=f"PDF Page {page_num+1}", width=400)

                        # Detect objects in the page
                        detections = detector.detect(image_np)
                        image_with_boxes, text_images, table_images, stamp_images, signature_images = process_image(
                            image_np,
                            detections,
                            ocr_processor,
                            page_num,
                            preprocessing_options,
                        )

                        st.image(
                            image_with_boxes,
                            caption=f"Image with Detections and Labels (Page {page_num+1})",
                            width=400,
                        )

                        # Display results sections
                        st.subheader(f"Extracted Entities (Page {page_num+1})")
                        entity_counter = 1

                        # Display confidence scores
                        st.write(f"## Confidence Scores (Page {page_num + 1}):")
                        with st.container():
                            confidence_dict = {}
                            for detection in detections:
                                if "class" in detection:
                                    confidence_dict[detection["class"]] = detection["confidence"]

                            for entity in ['text', 'table', 'stamp', 'signature']:
                                st.write(f"{entity_counter}) {entity.capitalize()}: {confidence_dict.get(entity, 'null')}")
                                entity_counter += 1

                        # Display detected entities
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

                        # Display extracted text
                        st.write("## Extracted Text:")
                        if text_images:
                            for detection in detections:
                                if "class" in detection and detection["class"] == "text":
                                    ocr_results = ocr_processor.process_detections(
                                        image_np, [detection], preprocessing_options
                                    )
                                    for result in ocr_results:
                                        st.write(f"Text: {result['text']}")
                        else:
                            st.write("No Text Detected")

            else:  # It's an image
                image = Image.open(uploaded_file).convert("RGB")
                image = np.array(image)
                st.image(image, caption="Uploaded Image", width=400)

                detections = detector.detect(image)
                image_with_boxes, text_images, table_images, stamp_images, signature_images = process_image(
                    image, detections, ocr_processor, preprocessing_options
                )

                st.image(
                    image_with_boxes, caption="Image with Detections and Labels", width=400
                )

                # Display results for image
                st.subheader("Extracted Entities")
                entity_counter = 1

                # Display confidence scores
                st.write("## Confidence Scores:")
                with st.container():
                    confidence_dict = {}
                    for detection in detections:
                        if "class" in detection:
                            confidence_dict[detection["class"]] = detection["confidence"]

                    for entity in ['text', 'table', 'stamp', 'signature']:
                        st.write(f"{entity_counter}) {entity.capitalize()}: {confidence_dict.get(entity, 'null')}")
                        entity_counter += 1

                # Display detected entities
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

                # Display extracted text
                st.write("## Extracted Text:")
                if text_images:
                    for detection in detections:
                        if "class" in detection and detection["class"] == "text":
                            ocr_results = ocr_processor.process_detections(
                                image, [detection], preprocessing_options
                            )
                            for result in ocr_results:
                                st.write(f"Text: {result['text']}")
                else:
                    st.write("No Text Detected")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.exception(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
