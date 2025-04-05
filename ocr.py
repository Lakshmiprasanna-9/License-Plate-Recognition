import cv2
import pytesseract
import streamlit as st

def extract_text(plate_image):
    # Convert to grayscale
    gray = cv2.cvtColor(plate_image, cv2.COLOR_RGB2GRAY)
    
    # **New: Apply morphological operations** to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Adaptive Threshold for clear text
    threshold = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    st.image(threshold, caption="Processed License Plate for OCR", use_container_width=True)  # Debugging

    # Use Tesseract to extract text
    text = pytesseract.image_to_string(threshold, config="--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    return text.strip()
