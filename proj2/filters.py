import cv2 as cv
import numpy as np
from scipy.ndimage import convolve1d

def convolution(img, kernel, padding_mode='constant', use_lib=False):
    flipped = np.flip(kernel)
    fh, fw = flipped.shape
    padded = np.pad(
        img, ((fh // 2, fh // 2), (fw // 2, fw // 2)), mode=padding_mode
    )
    result = np.empty(img.shape, dtype=np.float64)

    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            window = padded[r:r + fh, c:c + fw]
            result[r, c] = np.sum(window * flipped)
    return result

def blur_2d(img, h=9, s=0):
    kern = cv.getGaussianKernel(h, s).ravel()
    horizontal = convolve1d(img, kern, axis=1, mode='mirror')
    return convolve1d(horizontal, kern, axis=0, mode='mirror')

def gaussian_filter(h=9, s=0):
    one_d = cv.getGaussianKernel(ksize=h, sigma=s).ravel()
    return np.outer(one_d, one_d)
