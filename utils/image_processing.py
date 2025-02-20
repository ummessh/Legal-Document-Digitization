import cv2
import numpy as np

def ensure_gray(image):
    """
    Ensures the input image is in grayscale format.
    If the image is already grayscale, it's returned as-is.
    If it's in color, it's converted to grayscale.
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def deskew_hough(image):
    """
    Applies Hough Line Transform to detect and correct skew in the image.
    """
    gray = ensure_gray(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    if lines is not None:
        angle = 0
        for rho, theta in lines[0]:
            if theta < np.pi/4 or theta > 3*np.pi/4:
                angle = theta
                break
        if angle != 0:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle * 180 / np.pi - 90, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
    return image

def preprocess_image(image, options=None):
    if options is None:
        options = {}
    
    gray = ensure_gray(image)
    logger.info(f"Preprocessing options: {options}")
    
    if options.get('apply_threshold', False):
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.info("Applied thresholding")
    
    if options.get('apply_deskew', False):
        gray = deskew_hough(gray)
        logger.info("Applied deskewing")
    
    if options.get('apply_denoise', False):
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        logger.info("Applied denoising")
    
    if options.get('apply_contrast', False):
        gray = cv2.equalizeHist(gray)
        logger.info("Applied contrast enhancement")
    
    return gray
