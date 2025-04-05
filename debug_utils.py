import cv2
import os

def save_debug_image(image, name, debug=True):
    if debug:
        if not os.path.exists('debug'):
            os.makedirs('debug')
        cv2.imwrite(f'debug/{name}.jpg', image)