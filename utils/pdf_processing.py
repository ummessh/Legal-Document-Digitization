# utils/pdf_processor.py

from PIL import Image
import numpy as np
import fitz
import streamlit as st
import logging
from typing import Optional, Generator, Tuple, List
from .image_processing import preprocess_image
from .ocr_processor import OCRProcessor

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    A class to handle all PDF processing operations with better memory management
    and error handling.
    """
    def __init__(self, dpi: int = 300, language: str = "eng+hin+mar", psm: int = 6):
        """
        Initialize the PDF processor with configuration parameters.
        
        Args:
            dpi: Resolution for PDF rendering
            language: Languages for OCR processing
            psm: Page segmentation mode for OCR
        """
        self.dpi = dpi
        self.ocr_processor = OCRProcessor(language=language, psm=psm)
        
    def process_page(self, page: fitz.Page, options: Optional[dict] = None) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Process a single PDF page with enhanced error handling and memory management.
        
        Args:
            page: PDF page object
            options: Dictionary of preprocessing options
            
        Returns:
            Tuple of (processed image array, error message if any)
        """
        try:
            # Create pixmap with specific DPI
            matrix = fitz.Matrix(self.dpi/72, self.dpi/72)
            pix = page.get_pixmap(matrix=matrix)
            
            try:
                # Convert to PIL Image with proper color handling
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Preprocess if options provided
                if options:
                    image = preprocess_image(image, options)
                
                # Convert to numpy array
                image_np = np.array(image)
                
                return image_np, None
                
            finally:
                # Ensure resources are freed
                pix = None
                image = None
                
        except Exception as e:
            logger.error(f"Error processing PDF page: {str(e)}")
            return None, str(e)

    def process_pdf(self, uploaded_file, options: Optional[dict] = None) -> Generator[Tuple[int, Optional[np.ndarray], Optional[str]], None, None]:
        """
        Process PDF with memory-efficient page-by-page handling and OCR.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            options: Dictionary of preprocessing options
            
        Yields:
            Tuple of (page number, processed image array, extracted text)
        """
        doc = None
        progress_bar = st.progress(0)
        
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            total_pages = doc.page_count
            
            for page_num in range(total_pages):
                # Update progress
                progress_bar.progress((page_num + 1) / total_pages)
                
                try:
                    # Process one page at a time
                    page = doc[page_num]
                    image_np, error = self.process_page(page, options)
                    
                    if error:
                        logger.error(f"Error on page {page_num + 1}: {error}")
                        yield page_num, None, None
                        continue
                    
                    # Extract text using OCR if image processing succeeded
                    text = None
                    if image_np is not None:
                        text = self.ocr_processor.extract_text(image_np)
                    
                    yield page_num, image_np, text
                    
                finally:
                    # Clear page from memory
                    page = None
                    
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            yield None, None, None
            
        finally:
            # Clean up resources
            if doc:
                doc.close()
            progress_bar.empty()

    def convert_to_images(self, uploaded_file) -> List[Image.Image]:
        """
        Convert PDF to list of PIL Images with proper resource management.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            List of PIL Image objects
        """
        doc = None
        images = []
        
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            
            for page in doc:
                try:
                    matrix = fitz.Matrix(self.dpi/72, self.dpi/72)
                    pix = page.get_pixmap(matrix=matrix)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
                except Exception as e:
                    logger.error(f"Error converting page to image: {str(e)}")
                finally:
                    pix = None
                    
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            return []
            
        finally:
            if doc:
                doc.close()

    def __del__(self):
        """Cleanup method to ensure resources are freed."""
        self.ocr_processor = None
