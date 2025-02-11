import streamlit as st
import numpy as np
import sqlite3
import json
from PIL import Image
from models.yolo_detector import YOLODetector
from ocr.ocr_processor import OCRProcessor
from utils.config import Config

# Initialize models
detector = YOLODetector(Config.model_path)
ocr_processor = OCRProcessor(language=Config.ocr_languages, psm=Config.ocr_psm)

st.title("Legal Document Digitization")

uploaded_file = st.file_uploader("Upload an Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Run YOLO detection
    detections = detector.detect(image)

    # OCR + Error Correction
    extracted_data = ocr_processor.process_detections(image, detections)

    # Display corrected text
    st.subheader("Corrected Extracted Text:")
    st.json(extracted_data)

    # Store in database
    conn = sqlite3.connect(Config.db_path)
    cursor = conn.cursor()
    
    for item in extracted_data:
        if item['text']:  # Only store valid text
            cursor.execute("INSERT INTO ocr_results (bbox, text) VALUES (?, ?)", (str(item['bbox']), item['text']))
    
    conn.commit()
    conn.close()

    st.success("Corrected text stored in database.")
