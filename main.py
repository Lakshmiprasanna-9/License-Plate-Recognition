import cv2
import numpy as np
import pytesseract

# (Windows users: Uncomment and adjust the Tesseract path if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def is_license_plate(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h  # License plates are wider than they are tall
    area = cv2.contourArea(contour)
    # Adjust these thresholds based on your image dimensions and expected plate size.
    return 1000 < area < 30000 and 2 < aspect_ratio < 6

def refine_plate_region(plate_img, intensity_threshold=100, margin=5):
    """
    Refine the candidate license plate image by analyzing row-wise average intensity.
    """
    gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    row_avg = np.mean(gray_plate, axis=1)
    valid_rows = np.where(row_avg > intensity_threshold)[0]
    if len(valid_rows) == 0:
        return plate_img
    last_valid_row = valid_rows[-1]
    bottom_crop = min(last_valid_row + margin, plate_img.shape[0])
    refined_plate = plate_img[0:bottom_crop, :]
    return refined_plate

# ---------------------------
# Main Detection Code
# ---------------------------
# Load the image
image = cv2.imread("test.jpg")
if image is None:
    print("Error: Image not found!")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Edge detection using Canny
edges = cv2.Canny(blurred, 50, 200)
# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours found: {len(contours)}")

# Filter contours to get candidate license plates
license_plate_contours = [cnt for cnt in contours if is_license_plate(cnt)]
print(f"Possible license plates found: {len(license_plate_contours)}")

if not license_plate_contours:
    print("⚠️ No valid license plate found.")
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# Evaluate each candidate with a quick OCR and pick the best
best_candidate = None
best_text = ""
for cnt in license_plate_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    candidate = image[y:y+h, x:x+w]
    # Preprocess candidate for quick OCR
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    candidate_thresh = cv2.threshold(candidate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ocr_result = pytesseract.image_to_string(candidate_thresh, config="--psm 7")
    print(f"Candidate at X:{x},Y:{y},W:{w},H:{h} gives OCR text: '{ocr_result.strip()}'")
    if len(ocr_result.strip()) > len(best_text):
        best_text = ocr_result.strip()
        best_candidate = cnt

if best_candidate is None:
    print("⚠️ No candidate produced any text.")
    exit()

x, y, w, h = cv2.boundingRect(best_candidate)
print(f"Selected candidate: X:{x}, Y:{y}, W:{w}, H:{h}")
candidate_plate = image[y:y+h, x:x+w]

# Refine candidate region (crop unwanted dark area)
refined_plate = refine_plate_region(candidate_plate, intensity_threshold=100, margin=5)

# --- New steps to improve OCR ---
# 1. Enlarge the refined plate image to increase resolution.
scale_factor = 2.0  # Increase size by 2x; adjust as needed.
resized_plate = cv2.resize(refined_plate, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

# 2. Convert to grayscale and apply histogram equalization for contrast.
gray_resized = cv2.cvtColor(resized_plate, cv2.COLOR_BGR2GRAY)
equalized_plate = cv2.equalizeHist(gray_resized)
# Optionally, apply thresholding after equalization.
final_plate = cv2.threshold(equalized_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# --- Run OCR on the processed plate image ---
custom_config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
final_text = pytesseract.image_to_string(final_plate, config=custom_config)
print("Final extracted OCR text from refined candidate:")
print(final_text.strip())

# Visualization
vis_image = image.copy()
cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
cv2.imshow("Original Detection", vis_image)
cv2.imwrite("original_detection.jpg", vis_image)
cv2.imshow("Refined & Processed License Plate", final_plate)
cv2.imwrite("detected_plate.jpg", final_plate)
cv2.waitKey(0)
cv2.destroyAllWindows()


