from transformers import pipeline
import pytesseract
import logging
from utils.database_handler import store_ocr_result

class OCRProcessor:
    def __init__(self, language="eng+hin+mar", psm=6):
        self.language = language
        self.psm = psm
        self.corrector = pipeline("text2text-generation", model="ai4bharat/IndicBART")

    def extract_text(self, image):
        """Extracts text from an image using Tesseract OCR."""
        config = f"--oem 3 --psm {self.psm} preserve_interword_spaces=1"
        text = pytesseract.image_to_string(image, config=config, lang=self.language)
        return text.strip()

    def correct_text(self, text):
        """Corrects OCR errors using IndicBART for Hindi, Marathi, and English."""
        if not text.strip():
            return text  # Return unchanged if empty

        prompt = f"Fix OCR errors in: {text}"
        corrected_output = self.corrector(prompt, max_length=512, truncation=True)
        corrected_text = corrected_output[0]['generated_text']
        
        return corrected_text

    def process_detections(self, image, detections):
        """Runs OCR on detected text regions and applies IndicBERT correction."""
        results = []
        for det in detections:
            x, y, w, h = det['bbox']
            cropped_img = image[y:y+h, x:x+w]
            raw_text = self.extract_text(cropped_img)
            bbox, extracted_text = detection["bbox"], detection["text"]
            # Store raw extracted text before correction
            store_ocr_result(bbox, extracted_text)        
        return results
