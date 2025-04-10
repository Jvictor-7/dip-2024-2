# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2
import numpy as np
from skimage import exposure

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    matched_img = np.zeros_like(source_img, dtype=np.float64)
    
    for channel in range(3):
        matched_channel = exposure.match_histograms(
            source_img[..., channel],
            reference_img[..., channel],
            channel_axis=None
        )
        matched_img[..., channel] = matched_channel

    return np.clip(matched_img, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    source = cv2.cvtColor(cv2.imread('tasks/task-07-histogram-matching/source.jpg'), cv2.COLOR_BGR2RGB)
    reference = cv2.cvtColor(cv2.imread('tasks/task-07-histogram-matching/reference.jpg'), cv2.COLOR_BGR2RGB)

    matched = match_histograms_rgb(source, reference)

    # Converter de RGB para BGR para exibir no OpenCV
    source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
    reference_bgr = cv2.cvtColor(reference, cv2.COLOR_RGB2BGR)
    matched_bgr = cv2.cvtColor(matched, cv2.COLOR_RGB2BGR)

    # Exibir as imagens
    cv2.imshow('Source Image', source_bgr)
    cv2.imshow('Reference Image', reference_bgr)
    cv2.imshow('Matched Image', matched_bgr)

    print("Pressione qualquer tecla para fechar as janelas...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
