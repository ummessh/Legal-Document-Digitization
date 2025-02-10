import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from ocr.ocr_processor import OCRProcessor
from models.yolo_detector import YOLODetector
from utils.config import Config  # Ensure Config is properly defined
import json
import sqlite3
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Initialize YOLO model
try:
    detector = YOLODetector(Config.model_path)
except Exception as e:
    st.error(f"Failed to initialize YOLO detector: {e}")
    st.stop()

# Initialize OCR processor
try:
    ocr_processor = OCRProcessor(language=Config.ocr_languages, psm=Config.ocr_psm)
except Exception as e:
    st.error(f"Failed to initialize OCR processor: {e}")
    st.stop()

# Load mT5 Model for Text Correction
try:
    mt5_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    mt5_tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
except Exception as e:
    st.error(f"Failed to load mT5 model: {e}")
    st.stop()

def apply_mt5_correction(text):
    """Apply mT5 text correction to the input text."""
    try:
        input_ids = mt5_tokenizer("Correct: " + text, return_tensors="pt").input_ids
        output_ids = mt5_model.generate(input_ids, max_length=512)
        return mt5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        st.warning(f"Text correction failed: {e}")
        return text  # Return original text if correction fails

# Streamlit UI
st.title("Legal Document Digitization")

uploaded_file = st.file_uploader("Upload an Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Run YOLO model for object detection
        detections = detector.detect(image)
        if not detections:
            st.warning("No objects detected in the image.")
        else:
            st.success(f"Detected {len(detections)} objects.")

        # Process detections for OCR
        extracted_data = ocr_processor.process_detections(image, detections)
        if not extracted_data:
            st.warning("No text or tables extracted.")
        else:
            st.success(f"Extracted {len(extracted_data)} items.")

        # Apply mT5 Correction to extracted text
        for item in extracted_data:
            if item['type'] in ['text', 'table']:
                item['corrected_text'] = apply_mt5_correction(item['text'])

        # Display corrected text
        st.subheader("Corrected Text:")
        st.json(extracted_data)

        # Save to SQLite Database
        try:
            conn = sqlite3.connect(Config.db_path)
            cursor = conn.cursor()
            for item in extracted_data:
                if item['type'] in ['text', 'table']:
                    cursor.execute(
                        "INSERT INTO ocr_results (bbox, text) VALUES (?, ?)",
                        (str(item['bbox']), item['corrected_text'])
            conn.commit()
            conn.close()
            st.success("Text successfully extracted, corrected, and stored in the database.")
        except Exception as e:
            st.error(f"Failed to save data to the database: {e}")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
