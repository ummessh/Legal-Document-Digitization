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

from utils.config import Config  # Make sure this path is correct
from utils.pdf_processing import process_pdf  # Make sure this path is correct
from utils.image_processing import preprocess_image  # Make sure this path is correct
from models.yolo_detector import YOLODetector  # Make sure this path is correct

# ... (rest of your imports, logging setup, and OCRProcessor class)

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
            st.write(f"Detections: {detections}")  # Display detections for inspection

            image_with_boxes = image.copy()  # Create a copy to draw on

            if detections:
                for detection in detections:
                    bbox = detection['bbox']
                    x, y, w, h = map(int, bbox)  # Convert to integers
                    cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle

                    # Classify (replace with your actual classification logic)
                    if 'class' in detection:
                        category = detection['class']
                    elif 'confidence' in detection: # If class info is not available, use confidence
                        confidence = detection['confidence']
                        if confidence > 0.8: # Example threshold - adjust as needed
                            category = "text"
                        else:
                            category = "unknown"
                    else:
                        category = "unknown"

                    # Add class label to the image with bounding box
                    cv2.putText(image_with_boxes, str(category), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # OCR and display based on category
                    if category == "text":
                        ocr_results = ocr_processor.process_detections(image, [detection]) # Process one detection at a time
                        for result in ocr_results:
                            st.write(f"Category: {category}, Text: {result['text']}")
                    elif category == "table":
                        st.write(f"Category: {category}")  # Add table processing logic here
                    elif category == "stamp":
                        st.write(f"Category: {category}")  # Add stamp processing logic here
                    elif category == "signature":
                        st.write(f"Category: {category}")  # Add signature processing logic here
                    else:
                        st.write(f"Category: {category} (Unknown)")

                st.image(image_with_boxes, caption="Image with Detections and Labels")  # Display with boxes and labels

            else:
                st.write("No detections found by YOLO.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.exception(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
