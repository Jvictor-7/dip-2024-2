'''
2. Visualize Individual Color Channels

Objective: Extract and display the Red, Green, and Blue channels of a color image as grayscale and pseudo-colored images.

Topics: Channel separation and visualization.

Bonus: Reconstruct the original image using the separated channels.

'''

import cv2 as cv
import numpy as np
import requests
from matplotlib import pyplot as plt

def show_channels(image, image_name):
    # channels of a color image as grayscale
    B, G, R = cv.split(image)

    cv.imshow(f"{image_name} - Blue channel (Grayscale)", B)
    cv.imshow(f"{image_name} - Green channel (Grayscale)", G)
    cv.imshow(f"{image_name} - Red channel (Grayscale)", R)

    # channels of a color image as pseudo-colored images
    zeros = np.zeros_like(B)

    blue_pseudo = cv.merge([B, zeros, zeros])
    green_pseudo = cv.merge([zeros, G, zeros])
    red_pseudo = cv.merge([zeros, zeros, R])

    cv.imshow(f"{image_name} - Blue channel (Pseudo-color)", blue_pseudo)
    cv.imshow(f"{image_name} - Green channel (Pseudo-color)", green_pseudo)
    cv.imshow(f"{image_name} - Red channel (Pseudo-color)", red_pseudo)

if __name__ == "__main__":
    img_nature = cv.imread("./img/strawberries.tif")

    cv.imshow("Original Image", img_nature)

    show_channels(img_nature, "Image Nature")

    cv.waitKey(0)
    cv.destroyAllWindows()