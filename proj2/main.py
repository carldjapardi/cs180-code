import numpy as np
from scipy.signal import convolve2d
import cv2 as cv
from pathlib import Path


box_filter = 1/81 * np.ones((9, 9))
diff_x_filter = np.array([[1, 0, -1]])
diff_y_filter = diff_x_filter.reshape(-1, 1)

def convolution(img, filter, use_sci = False):
    if use_sci:
        return convolve2d(img, filter, mode='same', boundary='fill', fillvalue=0)
    filter = np.flip(filter)
    fh, fw = filter.shape
    pad_h, pad_w = fh//2, fw//2
    img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)))
    out_mat = np.zeros((img.shape[0], img.shape[1]))
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            inp_mat = img_padded[r:r+fh, c:c+fw]
            out_mat[r, c] = np.sum(inp_mat * filter) # ele wise mul
    return out_mat

def process_image_p11(file, filter, save_path, signed=False):
    im = cv.imread(file, cv.IMREAD_GRAYSCALE)
    im = im.astype(np.float32) / 255

    res = convolution(im, filter)
    res_scipy = convolution(im, filter, use_sci=True)

    if signed:
        scale = max(np.max(np.abs(res)), np.max(np.abs(res_scipy)))
        output = np.clip((res / scale + 1) * 127.5, 0, 255).astype(np.uint8)
        output_scipy = np.clip((res_scipy / scale + 1) * 127.5, 0, 255).astype(np.uint8)
    else:
        output = np.clip(res * 255, 0, 255).astype(np.uint8)
        output_scipy = np.clip(res_scipy * 255, 0, 255).astype(np.uint8)

    cv.imwrite(f'180-code/proj2/out/{save_path}.png', output)
    cv.imwrite(f'180-code/proj2/out/{save_path}_scipy.png', output_scipy)

def show_edges(file, threshold=0.35):
    im = cv.imread(file, cv.IMREAD_GRAYSCALE)
    im = im.astype(np.float32) / 255

    dx = convolution(im, diff_x_filter, use_sci=True)
    dy = convolution(im, diff_y_filter, use_sci=True)
    grad_magnitude = np.sqrt(dx**2 + dy**2)
    edges = grad_magnitude > threshold

    scale = max(np.max(np.abs(dx)), np.max(np.abs(dy)))
    dx_img = np.clip((dx / scale + 1) * 127.5, 0, 255).astype(np.uint8)
    dy_img = np.clip((dy / scale + 1) * 127.5, 0, 255).astype(np.uint8)
    grad_img = np.clip(grad_magnitude / scale * 255, 0, 255).astype(np.uint8)
    edges_img = (edges * 255).astype(np.uint8)
    
    out_dir = Path(__file__).resolve().parent / 'out'
    out_dir.mkdir(exist_ok=True)
    cv.imwrite(str(out_dir / '1_2_dx.png'), dx_img)
    cv.imwrite(str(out_dir / '1_2_dy.png'), dy_img)
    cv.imwrite(str(out_dir / '1_2_grad.png'), grad_img)
    cv.imwrite(str(out_dir / '1_2_edges.png'), edges_img)
    return edges

def show_edges_gaus(file):
    gaus_1d = cv.getGaussianKernel(ksize=9, sigma=0)
    gaus_2d_filt = np.outer(gaus_1d, gaus_1d.T)
    im = cv.imread(file, cv.IMREAD_GRAYSCALE)
    im = im.astype(np.float32) / 255
    im = convolution(im, gaus_2d_filt)
    dx = convolution(im, diff_x_filter, use_sci=True)
    dy = convolution(im, diff_y_filter, use_sci=True)
    grad_magnitude = np.sqrt(dx**2 + dy**2)
    edges = grad_magnitude > 0.1

    scale = max(np.max(np.abs(dx)), np.max(np.abs(dy)))
    dx_img = np.clip((dx / scale + 1) * 127.5, 0, 255).astype(np.uint8)
    dy_img = np.clip((dy / scale + 1) * 127.5, 0, 255).astype(np.uint8)
    grad_img = np.clip(grad_magnitude / scale * 255, 0, 255).astype(np.uint8)
    edges_img = (edges * 255).astype(np.uint8)
    
    out_dir = Path(__file__).resolve().parent / 'out'
    out_dir.mkdir(exist_ok=True)
    cv.imwrite(str(out_dir / '1_3_dx.png'), dx_img)
    cv.imwrite(str(out_dir / '1_3_dy.png'), dy_img)
    cv.imwrite(str(out_dir / '1_3_grad.png'), grad_img)
    cv.imwrite(str(out_dir / '1_3_edges.png'), edges_img)
    return edges

img1 = '180-code/proj2/selfie_.png'
# process_image_p11(img1, box_filter, '1_1_box')
# process_image_p11(img1, diff_x_filter, '1_1_x', signed=True)
# process_image_p11(img1, diff_y_filter, '1_1_y', signed=True)
img2 = '180-code/proj2/cameraman.png'
# show_edges(img2)
show_edges_gaus(img2)
