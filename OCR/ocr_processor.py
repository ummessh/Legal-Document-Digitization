import pytesseract
import json

class OCRProcessor:
    def __init__(self, language='eng+hin+mar', psm=6):
        """
        Initializes the OCR processor with specified languages and PSM mode.
        """
        self.language = language
        self.psm = psm

    def extract_text(self, image):
        """
        Extracts text from the given image using PyTesseract.
        """
        config = f"--oem 3 --psm {self.psm}"
        text = pytesseract.image_to_string(image, config=config, lang=self.language)
        return text.strip()

    def process_detections(self, image, detections):
        """
        Processes YOLO detections and applies OCR where needed.
        """
        extracted_data = []

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            roi = image[y1:y2, x1:x2]

            if det['class'] in [0, 1]:  # Text or Table
                text = self.extract_text(roi)
                extracted_data.append({'type': 'text' if det['class'] == 0 else 'table', 'bbox': det['bbox'], 'text': text})
            else:  # Signatures or Stamps
                extracted_data.append({'type': 'signature' if det['class'] == 2 else 'stamp', 'bbox': det['bbox'], 'present': True})

        return json.dumps(extracted_data, indent=2)
