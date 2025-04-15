'''
3. Convert Between Color Spaces (RGB ↔ HSV)

Objective: Convert an RGB image to other color spaces and display the result.

Topics: Color space conversion.

Challenge: Display individual channels from each converted space.
'''

import cv2 as cv
import numpy as np
import requests
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img_nature = cv.imread("./img/strawberries.tif")

    hsv = cv.cvtColor(img_nature, cv.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)

    mask = cv.bitwise_or(mask1, mask2)

    res = cv.bitwise_and(img_nature, img_nature, mask=mask)
    
    cv.imshow("Original", img_nature)
    cv.imshow("HSV Mask (Red)", mask)
    cv.imshow("Segment Red Areas", res)

    h, s, v = cv.split(hsv)
    cv.imshow("Hue", h)
    cv.imshow("Saturation", s)
    cv.imshow("Value", v)
    
    cv.waitKey(0)
    cv.destroyAllWindows()