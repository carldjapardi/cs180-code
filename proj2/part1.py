import numpy as np
from scipy.signal import convolve2d

from filters import convolution, gaussian_filter
from image_io import load_grayscale, save_image


box_filter = np.ones((9, 9)) / 81
diff_x_filter = np.array([[1, 0, -1]])
diff_y_filter = diff_x_filter.reshape(-1, 1)


def convolve_scipy(img, kernel):
    return convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0)


def process_image_p11(file, kernel, name, signed=False):
    img = load_grayscale(file)
    result = convolution(img, kernel)
    reference = convolve_scipy(img, kernel)

    limit = max(np.max(np.abs(result)), np.max(np.abs(reference))) if signed else 1.0
    save_image(name, result, signed=signed, limit=limit)
    save_image(f'{name}_scipy', reference, signed=signed, limit=limit)
    return result, reference


def save_edge_results(prefix, dx, dy, threshold, display_limit=None, suffix=''):
    magnitude = np.sqrt(dx**2 + dy**2)
    edges = magnitude > threshold
    if display_limit is None:
        display_limit = max(np.max(np.abs(dx)), np.max(np.abs(dy)))

    save_image(f'{prefix}_dx{suffix}', dx, signed=True, limit=display_limit)
    save_image(f'{prefix}_dy{suffix}', dy, signed=True, limit=display_limit)
    save_image(f'{prefix}_grad{suffix}', magnitude, limit=display_limit)
    save_image(f'{prefix}_edges{suffix}', edges.astype(float))
    return dx, dy, magnitude, edges


def show_edges(file, threshold=0.35):
    img = load_grayscale(file)
    dx = convolve_scipy(img, diff_x_filter)
    dy = convolve_scipy(img, diff_y_filter)
    return save_edge_results('1_2', dx, dy, threshold)


def show_edges_gaus(file, threshold=0.1):
    blurred = convolution(load_grayscale(file), gaussian_filter())
    save_image('1_3_blurred', blurred)
    dx = convolve_scipy(blurred, diff_x_filter)
    dy = convolve_scipy(blurred, diff_y_filter)
    return save_edge_results('1_3', dx, dy, threshold, display_limit=0.5)


def show_edges_gaus_single(file, threshold=0.1):
    gaussian = gaussian_filter()
    dog_x = convolve2d(gaussian, diff_x_filter, mode='full')
    dog_y = convolve2d(gaussian, diff_y_filter, mode='full')

    filter_limit = max(np.max(np.abs(dog_x)), np.max(np.abs(dog_y)))
    save_image('1_3_dog_x', dog_x, signed=True, limit=filter_limit)
    save_image('1_3_dog_y', dog_y, signed=True, limit=filter_limit)

    img = load_grayscale(file)
    dx = convolve_scipy(img, dog_x)
    dy = convolve_scipy(img, dog_y)
    return save_edge_results('1_3', dx, dy, threshold, display_limit=0.5, suffix='_conv1')
