import streamlit as st
import cv2
import numpy as np
import pytesseract
from detection import detect_license_plate
from ocr import extract_text

# Streamlit UI
st.title("License Plate Recognition System")
st.write("Upload an image to extract the license plate number.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Show uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Detect the license plate
    plate_image = detect_license_plate(image)

    if plate_image is not None:
        st.image(plate_image, caption="Detected License Plate", use_container_width=True)

        # Extract text from the plate
        extracted_text = extract_text(plate_image)

        if extracted_text:
            st.success(f"Extracted License Plate Number: **{extracted_text}**")
            print("Final extracted OCR text from refined candidate:")
            print(extracted_text.strip())  # Matches terminal output
        else:
            st.warning("⚠️ No text detected on the license plate.")
    else:
        st.warning("⚠️ No license plate detected. Try another image.")
