import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from ocr.ocr_processor import OCRProcessor
from models.yolo_detector import YOLODetector
from utils.config import Config
import json
import sqlite3
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Initialize YOLO model
detector = YOLODetector(Config.model_path)
ocr_processor = OCRProcessor(language=Config.ocr_languages, psm=Config.ocr_psm)

# Load mT5 Model for Text Correction
mt5_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
mt5_tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

def apply_mt5_correction(text):
    input_ids = mt5_tokenizer("Correct: " + text, return_tensors="pt").input_ids
    output_ids = mt5_model.generate(input_ids, max_length=512)
    return mt5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

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
    
    # Apply mT5 Correction
    for item in extracted_data:
        if item['type'] in ['text', 'table']:
            item['corrected_text'] = apply_mt5_correction(item['text'])
    
    # Display corrected text
    st.subheader("Corrected Text:")
    st.json(extracted_data)
    
    # Save to SQLite Database
    conn = sqlite3.connect(Config.db_path)
    cursor = conn.cursor()
    
    for item in extracted_data:
        if item['type'] in ['text', 'table']:
            cursor.execute("INSERT INTO ocr_results (bbox, text) VALUES (?, ?)", (str(item['bbox']), item['corrected_text']))
    
    conn.commit()
    conn.close()
    
    st.success("Text successfully extracted, corrected, and stored in database.")
