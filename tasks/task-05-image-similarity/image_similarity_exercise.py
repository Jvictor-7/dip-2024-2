# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    def mse(i1, i2):
        return np.mean((i1 - i2) ** 2)
    
    def psnr(i1, i2):
        mse_value = mse(i1, i2)
        
        if mse_value == 0:
            return float("inf")
        
        max_pixel = 1.0
        return 10 * np.log10(max_pixel ** 2 / mse_value)
    
    def ssim(i1, i2):
        C1 = 1e-4
        C2 = 9e-4
        
        mu_x = np.mean(i1)
        mu_y = np.mean(i2)
        sigma_x = np.var(i1)
        sigma_y = np.var(i2)
        sigma_xy = np.cov(i1.flatten(), i2.flatten())[0, 1]
        
        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        
        return numerator / denominator        
    
    def npcc(i1, i2):
        covariance = np.cov(i1.flatten(), i2.flatten())[0, 1]
        std_x = np.std(i1)
        std_y = np.std(i2)
        
        if std_x == 0 or std_y == 0:
            return 0.0
        
        return covariance / (std_x * std_y)
    
    mse_value = mse(i1, i2)
    psnr_value = psnr(i1, i2)
    ssim_value = ssim(i1, i2)
    npcc_value = npcc(i1, i2)
    
    return {
        "mse": mse_value,
        "psnr": psnr_value,
        "ssim": ssim_value,
        "npcc": npcc_value
    }