'''
1. Display Color Histograms for RGB Images

Objective: Calculate and display separate histograms for the R, G, and B channels of a color image.

Topics: Color histograms, channel separation.

Challenge: Compare histograms of different images (e.g., nature vs. synthetic images).
'''

import cv2 as cv
import numpy as np
import requests
from matplotlib import pyplot as plt

def plot_histogram(image, title, mask=None):
    chans = cv.split(image)
    colors = ("b", "g", "r")
    channel_names = ("Blue", "Green", "Red")
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of pixels")

    for chan, color, name in zip(chans, colors, channel_names):
        hist = cv.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0,256])
    
    plt.legend()

if __name__ == "__main__":
    img_nature = cv.imread("./img/strawberries.tif")
    img_synthetic = cv.imread("./img/rgbcube_kBKG.png")

    plot_histogram(img_nature, "Nature Image Histogram")
    plot_histogram(img_synthetic, "Synthetic Image Histogram") 

    cv.imshow("Img Nature", img_nature)
    cv.imshow("Img Synthetic", img_synthetic)

    plt.show()
  
    cv.waitKey(0)
    cv.destroyAllWindows()