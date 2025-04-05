import cv2
import numpy as np

def preprocess_image(image_path):
    # ✅ Ensure image_path is a valid string
    if not isinstance(image_path, str):
        raise ValueError(f"Invalid image path: {image_path}")

    image = cv2.imread(image_path)

    # ✅ Ensure image is loaded correctly
    if image is None:
        raise ValueError(f"Error: Unable to load image from {image_path}")

    # Convert grayscale images to RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur and edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    return edges
