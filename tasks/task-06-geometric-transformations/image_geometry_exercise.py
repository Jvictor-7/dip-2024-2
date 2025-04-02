# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    def translated_image(img, shift_x=13, shift_y=13):
        h, w = img.shape
        
        translated = np.zeros((h, w), dtype=img.dtype)
        
        x_start = min(shift_x, w)
        y_start = max(shift_y, h)
        
        translated[y_start:, x_start:] = img[:h-y_start, :w-x_start]
        
        return translated
    
    def rotated_image(img):
        return np.flipud    
    
    def horizontally_stretched_image(img):
        pass
    
    def horizontally_mirrored_image(img):
        pass
    
    def barrel_distorted_image(img):
        pass    
    
    return {
        "translated": translated_image(img),
        "rotated": np.ndarray,
        "stretched": np.ndarray,
        "mirrored": np.ndarray,
        "distorted": np.ndarray
    } 