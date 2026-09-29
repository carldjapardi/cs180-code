import numpy as np

from filters import convolution, gaussian_filter
from image_io import load_color, save_image


def unsharp_mask_filter(file, name, const=1):
    r, g, b = load_color(file)
    iden = np.zeros((9, 9))
    iden[4, 4] = 1
    filt = (1 + const) * iden - const * gaussian_filter()

    blurred = np.dstack([convolution(i, gaussian_filter()) for i in (r, g, b)])
    high_freq = np.dstack((r, g, b)) - blurred
    sharpened = np.dstack([convolution(i, filt) for i in (r, g, b)])

    save_image(f'2_1_1_{name}_blur', np.clip(blurred, 0, 1))
    save_image(f'2_1_1_{name}_high_freq', np.clip(high_freq + 0.5, 0, 1))
    save_image(f'2_1_1_{name}_sharp', np.clip(sharpened, 0, 1))


def sharp_to_blur_to_sharp(file, const=1):
    r, g, b = load_color(file)
    r_blur = convolution(r, gaussian_filter())
    g_blur = convolution(g, gaussian_filter())
    b_blur = convolution(b, gaussian_filter())

    iden = np.zeros((9, 9))
    iden[4, 4] = 1
    kernel = (1 + const) * iden - const * gaussian_filter()

    r_blur_unsharp = convolution(r_blur, kernel)
    g_blur_unsharp = convolution(g_blur, kernel)
    b_blur_unsharp = convolution(b_blur, kernel)

    unsharp = np.dstack((r_blur_unsharp, g_blur_unsharp, b_blur_unsharp))
    result = np.clip(unsharp, 0, 1)
    save_image('2_1_2_blur_sharp', result)
