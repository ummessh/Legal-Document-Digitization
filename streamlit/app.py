import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from OCR.ocr_processor import OCRProcessor
from models.yolo_detector import YOLODetector
from utils.config import Config
import json
import sqlite3

# Initialize YOLO model
detector = YOLODetector(Config.model_path)
ocr_processor = OCRProcessor(language=Config.ocr_languages, psm=Config.ocr_psm)

# Streamlit UI
st.title("Legal Document Digitization")

uploaded_file = st.file_uploader("Upload an Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Run YOLO model
    detections = detector.detect(image)
    
    # Process detections for OCR
    extracted_data = ocr_processor.process_detections(image, detections)
    
    # Placeholder for Spell Correction & NER (To be implemented later)
    # corrected_data = apply_spell_correction(extracted_data)
    
    # Display extracted text
    st.subheader("Extracted Text:")
    st.json(extracted_data)
    
    # Save to SQLite Database
    conn = sqlite3.connect(Config.db_path)
    cursor = conn.cursor()
    
    for item in json.loads(extracted_data):
        if item['type'] in ['text', 'table']:
            cursor.execute("INSERT INTO ocr_results (bbox, text) VALUES (?, ?)", (str(item['bbox']), item['text']))
    
    conn.commit()
    conn.close()
    
    st.success("Text successfully extracted and stored in database.")
