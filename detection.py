import cv2
import numpy as np
import streamlit as st

def detect_license_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Debug: Draw all contours found
    debug_img = image.copy()
    cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
    st.image(debug_img, caption="Contours Found", use_container_width=True)  # Show detected contours

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(c)

        # Debugging: Show properties of each contour
        print(f"Contour: x={x}, y={y}, w={w}, h={h}, Aspect Ratio={aspect_ratio}, Area={area}")

        # **Adjust aspect ratio & area conditions**
        if 300 < area < 50000 and 1.5 < aspect_ratio < 6:
            plate_img = image[y:y+h, x:x+w]
            return plate_img

    return None
