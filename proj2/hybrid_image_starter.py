import matplotlib.pyplot as plt
from align_image_code import align_images
from pathlib import Path
import numpy as np
import cv2 as cv
from filters import convolution, gaussian_filter

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / 'input'
OUT_DIR = PROJECT_DIR / 'out'

def impulse(h=9):
    mat = np.zeros((h, h))
    mat[h//2, h//2] = 1
    return mat

def hybrid_image(im1, im2, s1, s2):
    h1 = 2*int(3*s1) + 1
    h2 = 2*int(3*s2) + 1
    grayscale = True
    if grayscale:
        im1_low_pass = convolution(im1, gaussian_filter(h=h1, s=s1))
        high_pass_filt = impulse(h=h2) - gaussian_filter(h=h2, s=s2)
        im2_high_pass = convolution(im2, high_pass_filt)
        valid_cat = (im2 > 0).astype(np.uint8)
        safe_cat = cv.erode(valid_cat, np.ones((h2, h2), dtype=np.uint8))
        im2_high_pass *= safe_cat
        result = (im1_low_pass)/1.25 + 1.25*im2_high_pass
        result = np.clip(result, 0, 1)
        return result, im1_low_pass, im2_high_pass
    else:
        return


def save_fourier_spectrum(image, path):
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image))))
    plt.imsave(path, spectrum, cmap='gray')


def create_hybrid_image(im1, im2, name, sigma1=4, sigma2=3, save=True, save_process=False):
    im1_gray = cv.imread(str(im1), cv.IMREAD_GRAYSCALE).astype(np.float32) / 255.
    im2_gray = cv.imread(str(im2), cv.IMREAD_GRAYSCALE).astype(np.float32) / 255.
    #im1_aligned, im2_aligned = im1_gray, im2_gray
    im2_aligned, im1_aligned = align_images(im2_gray, im1_gray)

    hybrid, low_p, high_p = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

    plt.imshow(hybrid, cmap='gray')
    plt.show()
    if save or save_process:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    if save:
        plt.imsave(OUT_DIR / f'2_2_{name}.png', hybrid, cmap='gray', vmin=0, vmax=1)
    if save_process:
        images = {'original_1': im1_gray, 'original_2': im2_gray, 'aligned_1': im1_aligned, 'aligned_2': im2_aligned, 'low_pass': low_p, 'high_pass': high_p}
        for label, image in images.items():
            if label == 'high_pass':
                limit = max(np.max(np.abs(image)), 1e-8)
                plt.imsave(OUT_DIR / f'2_2_{name}_{label}.png', image,
                           cmap='gray', vmin=-limit, vmax=limit)
            else:
                plt.imsave(OUT_DIR / f'2_2_{name}_{label}.png', image,
                           cmap='gray', vmin=0, vmax=1)

        for label, image in (('aligned_1', im1_aligned),('aligned_2', im2_aligned),('low_pass', low_p),('high_pass', high_p),('hybrid', hybrid),):
            save_fourier_spectrum(image, OUT_DIR / f'2_2_{name}_fft_{label}.png')
    return hybrid, low_p, high_p 

if __name__ == '__main__':
    pass
