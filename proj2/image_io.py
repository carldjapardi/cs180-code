from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent / 'out'


def load_grayscale(file):
    img = cv.imread(str(file), cv.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255


def load_color(file):
    img = cv.imread(str(file), cv.IMREAD_COLOR)
    img = img.astype(np.float32) / 255
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return r, g, b


def save_image(name, data, signed=False, limit=1.0):
    OUT_DIR.mkdir(exist_ok=True)
    plt.imsave(
        OUT_DIR / f'{name}.png', data,
        cmap='gray',
        vmin=-limit if signed else 0,
        vmax=limit,
    )
