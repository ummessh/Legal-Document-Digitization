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
import pandas as pd  # Import pandas for table processing

from utils.config import Config  # Make sure this path is correct
from utils.pdf_processing import process_pdf  # Make sure this path is correct
from utils.image_processing import preprocess_image  # Make sure this path is correct
from models.yolo_detector import YOLODetector  # Make sure this path is correct

# Setup page configuration
st.set_page_config(
    page_title="Legal Document Digitization with YOLO OCR",
    page_icon=":page_facing_up:",
    layout="wide"
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# OCRProcessor class (Now only uses Tesseract)
class OCRProcessor:
    # ... (OCRProcessor class definition - no changes needed)


# Initialize models with improved caching
@st.cache_resource(max_entries=1)
def load_detector():
    # ... (load_detector function - no changes needed)

@st.cache_resource(max_entries=1)
def load_ocr_processor():
    # ... (load_ocr_processor function - no changes needed)


def main():
    detector = load_detector()
    ocr_processor = load_ocr_processor()

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")  # Ensure RGB format
            image = np.array(image)
            st.image(image, caption="Uploaded Image")

            detections = detector.detect(image)
            st.write(f"Detections: {detections}")

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

                    if 'class' in detection:  # Replace 'class' with your actual class label key
                        category = detection['class']
                    elif 'confidence' in detection:  # If no class label, use confidence
                        confidence = detection['confidence']
                        if confidence > 0.8:  # Adjust threshold as needed
                            category = "text"  # Or determine based on other detection properties
                        else:
                            category = "unknown"
                    else:
                        category = "unknown"

                    cv2.putText(image_with_boxes, str(category), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    roi = image[y:y+h, x:x+w]  # Extract ROI *here*

                    if category == "text":
                        ocr_results = ocr_processor.process_detections(image, [detection])
                        for result in ocr_results:
                            st.write(f"Category: {category}, Text: {result['text']}")
                        text_images.append(roi)

                    elif category == "table":
                        try:
                            # Example: If your table ROI is an image:
                            # img_bytes = cv2.imencode('.png', roi)[1].tobytes()
                            # df = pd.read_csv(io.BytesIO(img_bytes))
                            # Example: If your table ROI is CSV data:
                            # df = pd.read_csv(io.StringIO(roi))
                            # Example: If your table ROI is HTML data:
                            # df = pd.read_html(roi)[0]
                            # Replace the example with your actual conversion
                            df = pd.DataFrame() # Placeholder - Replace with your conversion
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

                st.image(image_with_boxes, caption="Image with Detections and Labels")

                st.subheader("Extracted Entities")

                if text_images:
                    st.write("Text:")
                    for i, img in enumerate(text_images):
                        st.write(f"{i+1})")  # Numbering
                        st.image(img)
                else:
                    st.write("Text: Not Detected")

                if table_images:
                    st.write("Tables:")
                    for i, img in enumerate(table_images):
                        st.write(f"{i+1})")  # Numbering
                        st.image(img)
                else:
                    st.write("Tables: Not Detected")

                if stamp_images:
                    st.write("Stamps:")
                    for i, img in enumerate(stamp_images):
                        st.write(f"{i+1})")  # Numbering
                        st.image(img)
                else:
                    st.write("Stamps: Not Detected")

                if signature_images:
                    st.write("Signatures:")
                    for i, img in enumerate(signature_images):
                        st.write(f"{i+1})")  # Numbering
                        st.image(img)
                else:
                    st.write("Signatures: Not Detected")

            else:
                st.write("No detections found by YOLO.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.exception(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
