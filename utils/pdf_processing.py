from PIL import Image
import numpy as np
import fitz
import streamlit as st
from .image_processing import preprocess_image
from .ocr_processor import OCRProcessor

# Initialize OCR Processor
ocr_processor = OCRProcessor(language="eng+hin+mar", psm=6)

def process_pdf_page(page, dpi=300, options=None):
    """Process a single PDF page with better error handling and resolution control."""
    try:
        # Create pixmap with specific DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        
        # Convert to PIL Image with proper color handling
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Preprocess the image if options are provided
        if options:
            image = preprocess_image(image, options)
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Clear memory
        pix = None
        image = None
        
        return image_np, None
    except Exception as e:
        return None, str(e)

def process_pdf(uploaded_file, options=None, progress_bar=None):
    """
    Process PDF with memory-efficient page-by-page handling.
    Includes OCR processing and text extraction.
    """
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        total_pages = doc.page_count
        extracted_text = []
        
        for page_num in range(total_pages):
            if progress_bar:
                progress_bar.progress((page_num + 1) / total_pages)
                
            # Process one page at a time
            page = doc[page_num]
            image_np, error = process_pdf_page(page, options=options)
            
            if error:
                st.error(f"Error processing page {page_num + 1}: {error}")
                continue
            
            # Extract text using OCR
            if image_np is not None:
                text = ocr_processor.extract_text(image_np)
                extracted_text.append(text)
                
            # Clear page from memory
            page = None
            
            yield page_num, image_np, text
            
        doc.close()
        
    except Exception as e:
        st.error(f"Error opening PDF: {e}")
        yield None, None, None

def convert_pdf_to_images(uploaded_file, dpi=300):
    """Utility function to convert PDF to list of images."""
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        images = []
        
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
            
        doc.close()
        return images
        
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []
