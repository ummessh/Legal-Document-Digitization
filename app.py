import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import logging
import sys
import io
import numpy as np
from PIL import Image
import fitz # PyMuPDF for PDF processing
import cv2 # OpenCV for image processing (used in mock preprocess_image and yolo_detector)
import pytesseract # For OCR (used in mock preprocess_image)
import requests # For Groq API calls (used in LLMchain)
import json # For JSON handling (used in LLMchain)

# LangChain imports (for LLMchain)
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Start of utils/config.py content ---
class Config:
    """
    Configuration settings for the Legal Document Digitization application.
    """
    # Placeholder for a YOLO model path.
    # In a real application, replace None with the actual path to your YOLOv8 model file.
    # Example: model_path = "models/yolov8n.pt"
    model_path = None # Set to None for the mock. For real usage, provide a valid path.
# --- End of utils/config.py content ---


# --- Start of models/yolo_detector.py content ---
class YOLODetector:
    def __init__(self, model_path=None):
        """
        Initializes the YOLO Detector.
        In a real application, this would load your YOLO model.
        :param model_path: Path to the YOLO model file (e.g., 'yolov8n.pt').
        """
        self.model_path = model_path if model_path is not None else Config.model_path
        if self.model_path:
            logger.info(f"YOLODetector initialized. Attempting to load model from: {self.model_path}")
            # Here you would load your actual YOLO model, e.g.:
            # from ultralytics import YOLO
            # self.model = YOLO(self.model_path)
            # For this mock, we don't load a real model.
        else:
            logger.warning("YOLODetector initialized without a specific model path. Using mock detections.")
        
    def detect(self, image_np: np.ndarray):
        """
        Performs object detection on a given image.
        This is a mock implementation that returns dummy detections.
        In a real application, this would run your YOLO model.

        :param image_np: The input image as a NumPy array (H, W, C).
        :return: A tuple (list of detection dictionaries, original image as NumPy array).
                 Each detection dictionary has 'bbox' (x, y, w, h) and 'class'.
        """
        if image_np is None:
            logger.error("YOLODetector received None for image_np. Cannot perform detection.")
            return [], None # Return empty detections if no image

        logger.info("Performing mock YOLO detection.")
        
        # For demonstration, create some dummy 'text' detections based on image dimensions
        h, w, _ = image_np.shape
        
        detections = []
        
        # Add a dummy text detection covering a significant part of the image
        text_bbox1 = [int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.3)] # x, y, width, height
        detections.append({
            'bbox': text_bbox1,
            'class': 'text',
            'confidence': 0.95
        })

        # Add another smaller dummy text detection
        text_bbox2 = [int(w * 0.2), int(h * 0.5), int(w * 0.6), int(h * 0.2)]
        detections.append({
            'bbox': text_bbox2,
            'class': 'text',
            'confidence': 0.90
        })

        # Example of other detection types (not processed by OCR here, but good for structure)
        table_bbox = [int(w * 0.15), int(h * 0.75), int(w * 0.5), int(h * 0.2)]
        detections.append({
            'bbox': table_bbox,
            'class': 'table',
            'confidence': 0.85
        })

        signature_bbox = [int(w * 0.7), int(h * 0.8), int(w * 0.2), int(h * 0.1)]
        detections.append({
            'bbox': signature_bbox,
            'class': 'signature',
            'confidence': 0.78
        })

        return detections, image_np # Return detections and the original image NumPy array
# --- End of models/yolo_detector.py content ---


# --- Start of utils/pdf_processing.py content (simplified as it's less used in the new flow) ---
# NOTE: The main app.py directly handles basic PDF page loading using fitz now.
# This function is kept for completeness but might not be explicitly called by the main logic.
def process_pdf(pdf_file, dpi=300):
    """
    Processes a PDF file and yields images for each page.
    This is a conceptual function as app.py now loads a single page directly.
    """
    logger.info("process_pdf function called (conceptual).")
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4: # CMYK to RGB conversion
            image_rgb = np.zeros((pix.height, pix.width, 3), dtype=np.uint8)
            image_rgb[:, :, 0] = image_np[:, :, 0] * (1 - image_np[:, :, 3] / 255.0)
            image_rgb[:, :, 1] = image_np[:, :, 1] * (1 - image_np[:, :, 3] / 255.0)
            image_rgb[:, :, 2] = image_np[:, :, 2] * (1 - image_np[:, :, 3] / 255.0)
            image_np = image_rgb
        yield image_np, page_num
    doc.close()
# --- End of utils/pdf_processing.py content ---


# --- Start of utils/image_processing.py content ---
def get_supported_languages_for_ocr():
    """Returns a dictionary of supported languages and their codes for Tesseract."""
    return {
        'English': 'eng',
        'Hindi': 'hin',
        'Marathi': 'mar'
    }

def preprocess_image(image_np: np.ndarray, detection: dict):
    """
    Extracts Region of Interest (ROI) from an image based on detection,
    applies basic preprocessing, and performs OCR using Tesseract.
    
    :param image_np: The full input image as a NumPy array (H, W, C).
    :param detection: A dictionary representing a single detection, expected to have a 'bbox' key.
                      Example: {'bbox': [x, y, w, h], 'class': 'text', ...}
    :return: A list of dictionaries, each with 'text' key for OCR results.
             Returns an empty list if no text can be extracted or bbox is invalid.
    """
    results = []
    
    if not isinstance(detection, dict) or 'bbox' not in detection:
        logger.error("Invalid detection object passed to preprocess_image. Missing 'bbox'.")
        return results

    bbox = detection['bbox']
    try:
        x, y, w, h = map(int, bbox)
        # Ensure bounding box coordinates are within image dimensions
        x = max(0, x)
        y = max(0, y)
        w = min(w, image_np.shape[1] - x)
        h = min(h, image_np.shape[0] - y)

        if w <= 0 or h <= 0:
            logger.warning(f"Invalid ROI dimensions for bbox {bbox}. Skipping OCR.")
            return results

        roi = image_np[y:y+h, x:x+w]

        # Convert ROI to PIL Image for Tesseract
        roi_pil = Image.fromarray(roi)

        # --- Tesseract OCR Configuration ---
        # For this combined file, using default English and PSM 3.
        # In a full app, you would take these from Streamlit sidebar options.
        lang_code = 'eng' # Default language
        psm = 3           # Default Page Segmentation Mode: Automatic page segmentation

        tesseract_config = f'-l {lang_code} --psm {psm}'
        
        text = ""
        try:
            text = pytesseract.image_to_string(roi_pil, config=tesseract_config)
            logger.info(f"OCR extracted text from ROI (bbox: {bbox}): '{text.strip()[:50]}...'")
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract is not installed or not in your PATH. Please install it.")
            text = "[Tesseract Not Found Error]"
        except Exception as e:
            logger.error(f"Error during Tesseract OCR on bbox {bbox}: {e}")
            text = f"[OCR Error: {e}]"

        results.append({
            'bbox': bbox,
            'text': text
        })

    except Exception as e:
        logger.error(f"Error processing detection bbox {bbox}: {e}")
        
    return results
# --- End of utils/image_processing.py content ---


# --- Start of models/LLMchain.py content ---
# It's good practice to get API keys securely from environment variables
# Ensure GROQ_API_KEY is set in your deployment environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

prompt_template = """
You are an expert in error correction and entity extraction, with expertise in multilingual processing (English, हिन्दी, मराठी).
Analyze the given text and perform the following tasks:
1. Named Entity Recognition: Identify key roles such as:
   - PERSON: Names of individuals (e.g., Mahesh, Suresh, etc.)
   - ORG: Organizations (Issuing Authority, Companies involved)
   - DATE: Important dates (Issue Date, Expiry Date, Agreement Date)
   - LOC: Locations mentioned in the document
   - OTHER: Any other relevant entities (e.g., Contract Number, Registration ID)
2. Summarization: Provide a brief summary of the document covering:
   - Document Type (Certificate, Agreement, Contract, etc.)
   - Purpose of the document
   - Key points (Validity, Terms, Clauses)

**Text:**
{text}

**IMPORTANT RULES**
1. The targeted domain of the text is legal documentation
2. CRITICAL: ALL output fields (document_type, summary, etc.) MUST be in the SAME LANGUAGE AND SCRIPT as the input text
3. If input is in Hindi script (देवनागरी), respond entirely in Hindi script
4. If input is in Marathi, respond entirely in Marathi
5. If input is in English, respond in English

Respond in this exact JSON format:
{{
    "entities Recognised": [
        {{
            "text": "extracted entity",
            "type": "entity type (PERSON, ORG, DATE, LOC, OTHER)"
        }}
    ],
    "document_type": "Detected document type (in same script as input)",
    "summary": "Brief summary of the document (in same script as input)"
}}
"""

class GroqLLM(LLM, BaseModel):
    # Ensure a default value or handle the case where GROQ_API_KEY might be None
    api_key: str = GROQ_API_KEY if GROQ_API_KEY else "" 
    model_name: str = "mixtral-8x7b-32768"
    temperature: float = 0.0
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if not self.api_key:
            return json.dumps({"error": "GROQ_API_KEY is not set. Please provide it as an environment variable."})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers
        )

        response_text = response.text
        # print("Raw API Response:", response_text)  # Keep for debugging if needed

        if response.status_code != 200:
            # Return a JSON string for consistency, which can be loaded by the caller
            return json.dumps({"error": f"Groq API error: {response.status_code} - {response_text}"})

        try:
            response_json = response.json()
            # Ensure 'content' exists before returning
            return response_json["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError) as e:
            return json.dumps({"error": f"Invalid JSON response or missing key from Groq API: {e}. Raw response: {response_text}"})

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

@st.cache_data(ttl=3600)  # Cache for 1 hour
def process_legal_text(text: str) -> Dict:
    try:
        MAX_CHARS = 10000 # max limit for text
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "..."

        if not GROQ_API_KEY:
            # Raise an error here, which will be caught by the outer try-except
            raise ValueError("GROQ_API_KEY is not found in environment variables. Please set it.")

        llm = GroqLLM(api_key=GROQ_API_KEY) # Pass the API key explicitly
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        
        # Using LCEL for chaining
        chain = prompt | llm 
        
        response_str = chain.invoke({"text": text})

        # Attempt to parse the response string as JSON
        response_dict = json.loads(response_str)

        # Check if the response dictionary itself contains an "error" key from GroqLLM._call
        if "error" in response_dict:
            # If there's an error from the LLM, raise it as an exception
            raise RuntimeError(response_dict["error"])

        return response_dict

    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse LLM response as JSON. Error: {str(e)}. Raw response: {response_str}"}
    except Exception as e:
        # Catch any other general exceptions during processing
        return {"error": f"Processing failed: {str(e)}"}
# --- End of models/LLMchain.py content ---


# --- Main Application Logic ---
# --- Database Setup ---
# Connect to SQLite database. check_same_thread=False is needed for Streamlit.
conn = sqlite3.connect("results.db", check_same_thread=False)
cursor = conn.cursor()

# Create table if it doesn't exist to store extraction results
cursor.execute("""
    CREATE TABLE IF NOT EXISTS extractions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        extracted_text TEXT,
        timestamp TEXT
    )
""")
conn.commit() # Commit changes to the database

# --- Streamlit App Start ---
st.set_page_config(page_title="Legal Document Digitization", layout="wide")
st.title("🚀 Legal Doc Digitizer + LLM Analysis")

uploaded_file = st.file_uploader("Upload PDF or Image", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    filename = uploaded_file.name
    file_bytes = uploaded_file.read()

    st.subheader("⚙️ Processing with YOLO v8")
    
    detector = YOLODetector() # Uses Config.model_path which is None in mock

    detections = []
    image_np_for_detection = None # Renamed to avoid conflict with `Image` import
    try:
        # Handle PDF and image files for initial loading into a processable image format
        if uploaded_file.type == "application/pdf":
            # For simplicity, process only the first page for detection in this combined mock.
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if doc.page_count > 0:
                page = doc.load_page(0)
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72)) # Render at 300 DPI
                image_np_for_detection = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4: # CMYK to RGB conversion if needed
                    image_rgb = np.zeros((pix.height, pix.width, 3), dtype=np.uint8)
                    image_rgb[:, :, 0] = image_np_for_detection[:, :, 0] * (1 - image_np_for_detection[:, :, 3] / 255.0)
                    image_rgb[:, :, 1] = image_np_for_detection[:, :, 1] * (1 - image_np_for_detection[:, :, 3] / 255.0)
                    image_rgb[:, :, 2] = image_np_for_detection[:, :, 2] * (1 - image_np_for_detection[:, :, 3] / 255.0)
                    image_np_for_detection = image_rgb
                st.info("Processing first page of PDF for detection.")
            else:
                st.warning("PDF has no pages to process.")
                image_np_for_detection = None
            doc.close()
        else:
            image_np_for_detection = np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"))

        if image_np_for_detection is not None:
            st.image(image_np_for_detection, caption="Uploaded Document (for detection)", use_column_width=True)
            detections, _ = detector.detect(image_np_for_detection) # Use the image_np here
            st.success("YOLO detection complete.")
        else:
            st.warning("Could not load image for detection.")

    except Exception as e:
        st.error(f"Error during document loading or YOLO detection: {e}")
        logger.exception("Error during document loading or YOLO detection")
        detections = []
        image_np_for_detection = None

    # Step 1: Extract text from detections
    combined_text = ""
    if detections and image_np_for_detection is not None:
        st.subheader("Text Extraction")
        for det in detections:
            if det.get("class") == "text":
                try:
                    ocr_results_for_det = preprocess_image(image_np_for_detection, det)
                    for r in ocr_results_for_det:
                        if "text" in r and r["text"].strip():
                            st.text_area(f"Extracted Text (bbox: {det['bbox']})", r["text"], height=100)
                            combined_text += r["text"] + "\n"
                except Exception as e:
                    st.warning(f"Error extracting text for a detection: {e}")
                    logger.exception("Error during text extraction for a detection")
    elif image_np_for_detection is None:
        st.error("No image could be processed for text extraction.")
    else:
        st.info("No text detections found in the document.")

    # Step 2: LLM analysis + save to database
    try:
        if combined_text.strip():
            st.subheader("🤖 LLM Analysis")
            with st.spinner("Analyzing text with LLM..."):
                llm_results = process_legal_text(combined_text)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT INTO extractions (filename, extracted_text, timestamp) VALUES (?, ?, ?)",
                (filename, combined_text, ts)
            )
            conn.commit()
            st.success("✅ Saved OCR text to database.")

            if llm_results and "error" not in llm_results:
                st.json(llm_results)
            else:
                st.warning(llm_results.get("error", "LLM couldn't process this text correctly."))
                logger.error(f"LLM processing failed: {llm_results.get('error', 'Unknown error')}")
        else:
            st.info("No textual content extracted for LLM analysis.")
    except Exception as e:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            cursor.execute(
                "INSERT INTO extractions (filename, extracted_text, timestamp) VALUES (?, ?, ?)",
                (filename, combined_text, ts)
            )
            conn.commit()
            st.warning("LLM processing failed, but extracted text was saved to database.")
        except Exception as db_e:
            st.error(f"Failed to save to database after LLM error: {db_e}")
        st.error(f"An error occurred during LLM analysis: {e}")
        logger.exception("Error during LLM analysis")

# --- Sidebar with Downloads ---
st.sidebar.header("📂 Saved OCR Entries")
try:
    df = pd.read_sql_query("SELECT * FROM extractions ORDER BY timestamp DESC", conn)
    if not df.empty:
        st.sidebar.dataframe(df)

        st.sidebar.download_button("⬇ Download CSV", df.to_csv(index=False), file_name="ocr_results.csv", mime="text/csv")
        
        txt_data = "\n\n".join([
            f"--- {row['filename']} [{row['timestamp']}] ---\n{row['extracted_text']}"
            for _, row in df.iterrows()
        ])
        st.sidebar.download_button("⬇ Download TXT", txt_data, file_name="ocr_results.txt", mime="text/plain")
        
        if os.path.exists("results.db"):
            with open("results.db", "rb") as f:
                st.sidebar.download_button("⬇ Download DB", f.read(), file_name="results.db", mime="application/octet-stream")
        else:
            st.sidebar.info("Database file not found.")
    else:
        st.sidebar.info("No OCR entries yet.")
except Exception as e:
    st.sidebar.error(f"Error retrieving saved OCR entries: {e}")
    logger.exception("Error retrieving saved OCR entries from database")
