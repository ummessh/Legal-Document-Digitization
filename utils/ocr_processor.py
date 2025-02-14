from transformers import pipeline
import pytesseract
import logging

class OCRProcessor:
    def __init__(self, language="eng+hin+mar", psm=6):
        self.language = language
        self.psm = psm
        self.corrector = None  # Default to None

        # Load IndicBART only if needed
        try:
            self.corrector = pipeline("text2text-generation", model="ai4bharat/IndicBART")
            logging.info("IndicBART model loaded successfully.")
        except Exception as e:
            logging.warning(f"Error loading IndicBART model: {e}")

    def get_supported_languages():
    """Returns a dictionary of supported languages and their codes."""
    return {
        'English': 'eng',
        'Hindi': 'hin',
        'Marathi':'mar'
    }

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
            logging.error(f"OCR error: {e}")
            return ""

    def correct_text(self, text):
        """Corrects OCR errors using IndicBART if available."""
        if not text.strip():
            return text

        if self.corrector is None:
            logging.warning("Skipping text correction: IndicBART model not loaded.")
            return text

        try:
            # Batch text correction for efficiency
            prompt = f"Fix OCR errors in: {text}"
            corrected_output = self.corrector(prompt, max_length=512, truncation=True)
            return corrected_output[0]['generated_text']
        except Exception as e:
            logging.error(f"IndicBART correction error: {e}")
            return text

    def process_detections(self, image, detections):
        """Runs OCR on detected text regions and applies text correction."""
        results = []
        for det in detections:
            try:
                x, y, w, h = map(int, det['bbox'])  # Convert bbox values safely
                cropped_img = image[y:y+h, x:x+w]

                raw_text = self.extract_text(cropped_img)
                corrected_text = self.correct_text(raw_text)

                results.append({
                    'bbox': det['bbox'],
                    'raw_text': raw_text,
                    'corrected_text': corrected_text,
                    'class': det.get('class', 'text')  # Default class to 'text'
                })
            except Exception as e:
                logging.error(f"Error processing detection {det}: {e}")

        return results
