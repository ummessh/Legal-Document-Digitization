from PIL import Image
import numpy as np
from .image_processing import preprocess_image
from .ocr_processor import OCRProcessor 

# Initialize OCR Processor
ocr_processor = OCRProcessor(language="eng+hin+mar", psm=6)

def process_pdf(uploaded_file, options):
    """
    Process an uploaded PDF file:
    1. Convert PDF to images
    2. Preprocess images
    3. Extract text using OCR
    """
    try:
        # Convert uploaded PDF file to images
        images = convert_pdf_to_images(uploaded_file)
        extracted_text = []
        for img in images:
            preprocessed_img = preprocess_image(img, options)
            text = ocr_processor.extract_text(preprocessed_img)  
            extracted_text.append(text)
        return extracted_text

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []
