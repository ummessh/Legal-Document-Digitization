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
import sqlite3
from datetime import datetime

from utils.config import Config 
from utils.pdf_processing import process_pdf
from utils.image_processing import preprocess_image
from models.yolo_detector import YOLODetector
# CHANGES
from models.LLMchain import process_legal_text
st.set_page_config(
    page_title="Legal Document Digitization with YOLO OCR",
    page_icon=":page_facing_up:",
    layout="wide"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def process_pdf_page(page, dpi=300):
    """
    Process a single PDF page and convert it to a numpy array.
    
    Args:
        page: fitz.Page object
        dpi: int, resolution for rendering (default: 300)
    
    Returns:
        tuple: (numpy array of the image, error message if any)
    """
    try:
        # Get the page's pixel matrix
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        
        # Convert to numpy array
        image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        
        # If the image is CMYK (4 channels), convert to RGB (3 channels)
        if pix.n == 4:
            # Create RGB image
            image_rgb = np.zeros((pix.height, pix.width, 3), dtype=np.uint8)
            # Simple CMYK to RGB conversion
            image_rgb[:, :, 0] = image_np[:, :, 0] * (1 - image_np[:, :, 3] / 255.0)
            image_rgb[:, :, 1] = image_np[:, :, 1] * (1 - image_np[:, :, 3] / 255.0)
            image_rgb[:, :, 2] = image_np[:, :, 2] * (1 - image_np[:, :, 3] / 255.0)
            image_np = image_rgb

        return image_np, None
        
    except Exception as e:
        return None, str(e)

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

# Database Setup
conn = sqlite3.connect("results.db", check_same_thread=False)
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    extracted_text TEXT,
    timestamp TEXT
)
""")
conn.commit()

    
def main():
    detector = load_detector()
    ocr_processor = load_ocr_processor()
    st.title("Legal Document Digitizer")
    st.write("By Aryan Tandon and Umesh Tiwari")

    # Sidebar for options
    st.sidebar.title("Document Processing Options")

    # View DB Section
    st.sidebar.subheader("📂 View Saved OCR Results")
    if st.sidebar.button("📄 Show Entries"):
        df = pd.read_sql_query("SELECT * FROM extractions ORDER BY timestamp DESC", conn)
        st.write("### Saved OCR Results:")
        st.dataframe(df)

        st.download_button("⬇ Download as CSV", df.to_csv(index=False), file_name="ocr_results.csv")
        txt_data = "\n\n".join([f"{row['filename']} ({row['timestamp']}):\n{row['extracted_text']}" for _, row in df.iterrows()])
        st.download_button("⬇ Download as TXT", txt_data, file_name="ocr_results.txt")

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
    apply_threshold = st.sidebar.checkbox(
        "Sharpen Text", value=True, help="Improves text clarity by increasing contrast"
    )
    apply_deskew = st.sidebar.checkbox(
        "Straighten Document", value=True, help="Corrects tilted or skewed documents"
    )
    apply_denoise = st.sidebar.checkbox(
        "Remove Background Noise",
        value=True,
        help="Removes specks and background interference",
    )
    apply_contrast = st.sidebar.checkbox(
        "Enhance Text Visibility", value=False, help="Boosts text brightness and contrast"
    )

    preprocessing_options = {
        "apply_threshold": apply_threshold,
        "apply_deskew": apply_deskew,
        "apply_denoise": apply_denoise,
        "apply_contrast": apply_contrast,
    }

    uploaded_file = st.file_uploader(
        "Choose an image or PDF...", type=["jpg", "png", "jpeg", "pdf"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                # Add a progress bar
                progress_bar = st.progress(0)

                try:
                    # Use the improved PDF processing
                    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    total_pages = doc.page_count

                    for page_num in range(total_pages):
                        # Update progress
                        progress_bar.progress((page_num + 1) / total_pages)

                        # Process one page at a time
                        page = doc[page_num]
                        image_np, error = process_pdf_page(page, dpi=300)

                        if error:
                            st.error(f"Error processing page {page_num + 1}: {error}")
                            continue

                        if image_np is None:
                            continue

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

                        st.subheader(f"Extracted Entities (Page {page_num+1})")
                        entity_counter = 1

                        # Display confidence scores
                        st.write(f"## Confidence Scores (Page {page_num + 1}):")
                        with st.container():
                            confidence_dict = {}
                            for detection in detections:
                                if "class" in detection:
                                    confidence_dict[detection["class"]] = detection["confidence"]

                            st.write(f"1) Text: {confidence_dict.get('text', 'null')}")
                            st.write(f"2) Table: {confidence_dict.get('table', 'null')}")
                            st.write(f"3) Stamp: {confidence_dict.get('stamp', 'null')}")
                            st.write(f"4) Signature: {confidence_dict.get('signature', 'null')}")

                        # Display detected entities
                        if text_images:
                            st.write("Text:")
                            for img in text_images:
                                st.write(f"{entity_counter})")
                                st.image(img, width=400)
                                entity_counter += 1
                        else:
                            st.write(f"{entity_counter}) Text: Not Detected")
                            entity_counter += 1

                        if table_images:
                            st.write("Tables:")
                            for img in table_images:
                                st.write(f"{entity_counter})")
                                st.image(img, width=400)
                                entity_counter += 1
                        else:
                            st.write(f"{entity_counter}) Tables: Not Detected")
                            entity_counter += 1

                        if stamp_images:
                            st.write("Stamps:")
                            for img in stamp_images:
                                st.write(f"{entity_counter})")
                                st.image(img, width=400)
                                entity_counter += 1
                        else:
                            st.write(f"{entity_counter}) Stamps: Not Detected")
                            entity_counter += 1

                        if signature_images:
                            st.write("Signatures:")
                            for img in signature_images:
                                st.write(f"{entity_counter})")
                                st.image(img, width=400)
                                entity_counter += 1
                        else:
                            st.write(f"{entity_counter}) Signatures: Not Detected")
                            entity_counter += 1
#CHANGES Start
                        st.write("## Extracted Text:")
                        if text_images:
                            combined_text = ""
                            for detection in detections:
                                if "class" in detection and detection["class"] == "text":
                                    ocr_results = ocr_processor.process_detections(image_np, [detection], preprocessing_options)
                                    for result in ocr_results:
                                        st.write(f"Text: {result['text']}")
                                        combined_text += result['text'] + "\n"
                            
                            if combined_text.strip():
                                st.subheader("LLM Analysis")
                                with st.spinner("Analyzing text with LLM..."):
                                    llm_results = process_legal_text(combined_text)
                                    if "error" not in llm_results:
                                        st.json(llm_results)
                                    else:
                                        st.error(llm_results["error"])
                        else:
                            st.write("No Text Detected")
#CHANGES End
                        # Clear lists for the next page
                        text_images = []
                        table_images = []
                        stamp_images = []
                        signature_images = []

                        # Clear page from memory
                        page = None

                    # Close the document
                    doc.close()

                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                    logger.exception(f"Error processing PDF: {e}")

                finally:
                    # Clear the progress bar
                    progress_bar.empty()

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

                st.subheader("Extracted Entities")
                entity_counter = 1

                st.write("## Confidence Scores:")
                with st.container():
                    confidence_dict = {}
                    for detection in detections:
                        if "class" in detection:
                            confidence_dict[detection["class"]] = detection["confidence"]

                    st.write(f"1) Text: {confidence_dict.get('text', 'null')}")
                    st.write(f"2) Table: {confidence_dict.get('table', 'null')}")
                    st.write(f"3) Stamp: {confidence_dict.get('stamp', 'null')}")
                    st.write(f"4) Signature: {confidence_dict.get('signature', 'null')}")

                if text_images:
                    st.write("Text:")
                    for img in text_images:
                        st.write(f"{entity_counter})")
                        st.image(img, width=400)
                        entity_counter += 1
                else:
                    st.write(f"{entity_counter}) Text: Not Detected")
                    entity_counter += 1

                if table_images:
                    st.write("Tables:")
                    for img in table_images:
                        st.write(f"{entity_counter})")
                        st.image(img, width=400)
                        entity_counter += 1
                else:
                    st.write(f"{entity_counter}) Tables: Not Detected")
                    entity_counter += 1

                if stamp_images:
                    st.write("Stamps:")
                    for img in stamp_images:
                        st.write(f"{entity_counter})")
                        st.image(img, width=400)
                        entity_counter += 1
                else:
                    st.write(f"{entity_counter}) Stamps: Not Detected")
                    entity_counter += 1

                if signature_images:
                    st.write("Signatures:")
                    for img in signature_images:
                        st.write(f"{entity_counter})")
                        st.image(img, width=400)
                        entity_counter += 1
                else:
                    st.write(f"{entity_counter}) Signatures: Not Detected")
                    entity_counter += 1

                try:#CHANGES start
                    st.write("## Extracted Text:")
                    combined_text = ""
                    if text_images:
                       
                        for detection in detections:
                            if "class" in detection and detection["class"] == "text":
                                ocr_results = ocr_processor.process_detections(image, [detection], preprocessing_options)
                                for result in ocr_results:
                                    st.write(f"Text: {result['text']}")
                                    combined_text += result['text'] + "\n"
                        
                    if combined_text.strip():
                        st.subheader("LLM Analysis")
                        with st.spinner("Analyzing text with LLM..."):
                            llm_results = process_legal_text(combined_text)
                            if "error" not in llm_results:
                                st.json(llm_results)
                            else:
                                st.error(llm_results["error"])

                            # After LLM processing
                            filename = uploaded_file.name + f"_page_{page_num+1}"
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            cursor.execute(
                                "INSERT INTO extractions (filename, extracted_text, timestamp) VALUES (?, ?, ?)",
                                (filename, combined_text, timestamp)
                            )
                            conn.commit()
                            st.success(f"✅ Page {page_num+1} saved to database.")

                        # ✅ Save to SQLite here
                        filename = uploaded_file.name
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cursor.execute("INSERT INTO extractions (filename, extracted_text, timestamp) VALUES (?, ?, ?)",(filename, combined_text, timestamp))
                        conn.commit()
                        st.success("✅ Text saved to database.")        
                    else:
                        st.write("No Text Detected")
#CHANGES end
                except Exception as e:
                    st.error(f"An error occurred during image text extraction: {e}")
                    logger.exception(f"An error occurred during image text extraction: {e}")

        except Exception as e:
            st.error(f"An outer error occurred: {e}")
            logger.exception(f"An outer error occurred: {e}")

if __name__ == "__main__":
    main()
