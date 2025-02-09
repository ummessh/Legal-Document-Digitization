class OCRConfig:
    """
    Configuration settings for the OCR processing module.
    """
    # Default OCR languages (can be modified based on user needs)
    LANGUAGES = "eng+hin+mar"
    
    # Page Segmentation Modes (PSM)
    # 6: Assume a single uniform block of text
    PSM = 6
    
    # Tesseract OCR Engine Mode (OEM)
    # 3: Default, using both LSTM and legacy models
    OEM = 3
    
    # Additional Tesseract configuration options
    TESSERACT_CONFIG = f"--oem {OEM} --psm {PSM} preserve_interword_spaces=1"
    
    @classmethod
    def get_config(cls):
        """Returns the OCR configuration string for Tesseract."""
        return cls.TESSERACT_CONFIG
