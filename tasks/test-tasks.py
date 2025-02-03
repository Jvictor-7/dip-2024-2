import numpy as np
import cv2

def generate_image(seed, width, height, mean, std):
    """
    Generates a grayscale image with pixel values sampled from a normal distribution.

    Args:
        seed (int): Random seed for reproducibility (student's registration number).
        width (int): Width of the generated image.
        height (int): Height of the generated image.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.

    Returns:
        image (numpy.ndarray): The generated image.
    """
    ### START CODE HERE ###
    np.random.seed(seed) 
    image = np.random.normal(loc=mean, scale=std, size=(height, width)) 
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    ### END CODE HERE ###

    return image

img = generate_image(seed=19212513, width=300, height=300, mean=128, std=50)

cv2.imshow('Generated Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 