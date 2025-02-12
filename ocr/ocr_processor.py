from transformers import pipeline
import pytesseract
import logging

class OCRProcessor:
    def __init__(self, language="eng+hin+mar", psm=6):
        self.language = language
        self.psm = psm
        try:
            self.corrector = pipeline("text2text-generation", model="ai4bharat/IndicBART")
        except Exception as e:
            logging.error(f"Error loading IndicBART model: {e}")
            self.corrector = None  # Set to None if loading fails

    def extract_text(self, image):
        """Extracts text from an image using Tesseract OCR."""
        config = f"--oem 3 --psm {self.psm} preserve_interword_spaces=1"
        try:
            text = pytesseract.image_to_string(image, config=config, lang=self.language)
            return text.strip()
        except pytesseract.TesseractNotFoundError:
            logging.error("Tesseract is not installed or not in your PATH.")
            return ""
        except Exception as e:
            logging.error(f"An error occurred during OCR: {e}")
            return ""

    def correct_text(self, text):
        """Corrects OCR errors using IndicBART."""
        if not text.strip():
            return text

        if self.corrector is None:
            logging.warning("IndicBART model not loaded. Skipping correction.")
            return text

        try:
            prompt = f"Fix OCR errors in: {text}"
            corrected_output = self.corrector(prompt, max_length=512, truncation=True)
            corrected_text = corrected_output[0]['generated_text']
            return corrected_text
        except Exception as e:
            logging.error(f"Error in IndicBART pipeline: {e}")
            return text

    def process_detections(self, image, detections):
        """Runs OCR and applies IndicBART correction. Returns results."""
        results = []
        for det in detections:
            x, y, w, h = map(int, det['bbox'])  # Convert to integers HERE
            cropped_img = image[y:y+h, x:x+w]
            raw_text = self.extract_text(cropped_img)
            corrected_text = self.correct_text(raw_text)

            results.append({
                'bbox': det['bbox'],
                'raw_text': raw_text,
                'corrected_text': corrected_text,
                'class': det.get('class', 'text') # Add class information. Default to 'text' if not available
            })

        return results
