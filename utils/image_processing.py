def preprocess_image(image, options=None):
    """
    Applies various preprocessing techniques to improve OCR accuracy:
    1. Converts to grayscale
    2. Optionally applies thresholding
    3. Optionally deskews the image
    4. Optionally applies denoising
    5. Optionally enhances contrast
    
    Parameters:
    - image: Input image to preprocess
    - options: Dictionary of preprocessing options, defaults to None
    
    Returns:
    - Preprocessed grayscale image
    """
    # Initialize options as empty dict if None
    if options is None:
        options = {}
    
    gray = ensure_gray(image)
    
    if options.get('apply_threshold', False):
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if options.get('apply_deskew', False):
        gray = deskew_hough(gray)
    
    if options.get('apply_denoise', False):
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    if options.get('apply_contrast', False):
        gray = cv2.equalizeHist(gray)
    
    return gray
