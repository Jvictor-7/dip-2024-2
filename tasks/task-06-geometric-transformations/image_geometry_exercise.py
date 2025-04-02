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
    h, w = img.shape
    
    def translated_image(img, shift_x=13, shift_y=13):
        
        translated = np.zeros((h, w), dtype=img.dtype)
        
        x_start = min(shift_x, w)
        y_start = min(shift_y, h)
        
        translated[y_start:h, x_start:w] = img[:h - y_start, :w - x_start]
        
        return translated
    
    def rotated_image(img):
        return np.rot90(img, k=3) ## or np.flipud(img.T)
    
    def horizontally_stretched_image(img, scale=1.5):
        new_w = int(w * scale)
        x_indices = np.linspace(0, w - 1, new_w)
        
        stretched = np.zeros((h, new_w), dtype=img.dtype)
        for i in range(h):
            stretched[i, :] = np.interp(x_indices, np.arange(w), img[i, :])
        
        return stretched
        
    def horizontally_mirrored_image(img):
        return img[:, ::-1]
    
    def barrel_distorted_image(img, distortion_factor=0.2):
        cx = w // 2
        cy = h // 2
        
        distorted = np.zeros_like(img) 
        
        y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        x_norm = (x_indices - cx) / cx
        y_norm = (y_indices - cy) / cy
        
        r = np.sqrt(x_norm ** 2 + y_norm ** 2)
        r_safe = np.where(r == 0, 1, r)
        
        r_distorted = r + distortion_factor * r ** 3 
        
        x_new = (cx * (x_norm / r_safe) * r_distorted + cx).astype(int)
        y_new = (cy * (y_norm / r_safe) * r_distorted + cy).astype(int)
        
        valid = (x_new >= 0) & (x_new < w) & (y_new >= 0) & (y_new < h)
        
        distorted[y_indices[valid], x_indices[valid]] = img[y_new[valid], x_new[valid]]
        
        return distorted        
    
    return {
        "translated": translated_image(img),
        "rotated": rotated_image(img),
        "stretched": horizontally_stretched_image(img),
        "mirrored": horizontally_mirrored_image(img),
        "distorted": barrel_distorted_image(img),
    }