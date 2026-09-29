"""Export the additional figures and measured checks used by the project report.

Run from any directory: python3 report_assets.py
Existing input files, processing code, and saved outputs are left in place.
"""
from pathlib import Path
from time import perf_counter
import json

import cv2 as cv
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import hsv_to_rgb
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

from filters import convolution, gaussian_filter
from image_io import load_grayscale
from part2_3 import gaus_lap_stack
from part2_4 import circular_mask_from_points

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / 'input'
ASSET_DIR = PROJECT_DIR.parents[1] / '180-portfolio' / 'proj2' / 'assets'


def convolution_four_loops(img, kernel):
    flipped = np.flip(kernel)
    fh, fw = flipped.shape
    padded = np.pad(img, ((fh // 2, fh // 2), (fw // 2, fw // 2)))
    result = np.zeros(img.shape, dtype=np.float64)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            for kr in range(fh):
                for kc in range(fw):
                    result[r, c] += padded[r + kr, c + kc] * flipped[kr, kc]
    return result


def rgb(name):
    image = cv.imread(str(INPUT_DIR / name), cv.IMREAD_COLOR)
    return cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32) / 255.


def save(name, image):
    pixels = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(ASSET_DIR / f'{name}.png', optimize=True)


def filter_rgb(image, kernel):
    return np.dstack([convolve2d(image[:, :, ch], kernel, mode='same')
                     for ch in range(3)])


def copy_images():
    for path in sorted((PROJECT_DIR / 'out').glob('*.png')):
        # These are untracked earlier experiments, not the final reproducible runs.
        if path.name in ('2_4.png', '2_4_crb.png'):
            continue
        image = Image.open(path)
        if image.mode == 'RGBA' and image.getextrema()[3] == (255, 255):
            image = image.convert('RGB')
        image.save(ASSET_DIR / path.name, optimize=True)
        exported = ASSET_DIR / path.name
        if exported.stat().st_size > 800_000:
            image.save(exported.with_suffix('.webp'), quality=92, method=6)
            exported.unlink()
    for path in sorted(INPUT_DIR.iterdir()):
        if path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
            image = Image.open(path).convert('RGB')
            image.thumbnail((1280, 1280))
            image.save(ASSET_DIR / f'original_{path.stem.lower()}.jpg', quality=90)
    save('selfie_gray', load_grayscale(INPUT_DIR / 'selfie_.png'))


def check_convolution():
    image = load_grayscale(INPUT_DIR / 'selfie_.png')[:128, :128]
    kernels = {'box': np.ones((9, 9)) / 81,
               'dx': np.array([[1, 0, -1]]),
               'dy': np.array([[1], [0], [-1]])}
    measurements = {}
    for name, kernel in kernels.items():
        results = {}
        timings = {}
        for label, fn in [('four', convolution_four_loops), ('two', convolution),
                          ('scipy', lambda a, k: convolve2d(a, k, mode='same'))]:
            elapsed = []
            for _ in range(3):
                start = perf_counter()
                results[label] = fn(image, kernel)
                elapsed.append((perf_counter() - start) * 1000)
            timings[label] = float(np.median(elapsed))
        measurements[name] = {
            'milliseconds': timings,
            'four_error': float(np.max(np.abs(results['four'] - results['scipy']))),
            'two_error': float(np.max(np.abs(results['two'] - results['scipy'])))
        }
    return measurements


def edges_and_orientation():
    image = load_grayscale(INPUT_DIR / 'cameraman.png')
    gaussian = gaussian_filter()
    dx_kernel = np.array([[1, 0, -1]])
    blurred = convolution(image, gaussian)
    dx = convolve2d(blurred, dx_kernel, mode='same')
    dy = convolve2d(blurred, dx_kernel.T, mode='same')
    dog_x = convolve2d(gaussian, dx_kernel, mode='full')
    dog_y = convolve2d(gaussian, dx_kernel.T, mode='full')
    direct_x = convolve2d(image, dog_x, mode='same')
    direct_y = convolve2d(image, dog_y, mode='same')
    save('gaussian_kernel', gaussian / gaussian.max())

    # Compute an angle from a slope without arctan, arctan2, or np.angle.
    # Range reduction keeps the alternating series argument within ±0.4143.
    ax, ay = np.abs(dx), np.abs(dy)
    z = np.minimum(ax, ay) / np.maximum(np.maximum(ax, ay), 1e-12)
    t = (z - (np.sqrt(2) - 1)) / (1 + z * (np.sqrt(2) - 1))
    angle = np.full(z.shape, np.pi / 8)
    for n in range(12):
        angle += (-1)**n * t**(2*n + 1) / (2*n + 1)
    angle = np.where(ay > ax, np.pi / 2 - angle, angle)
    angle = np.where(dx < 0, np.pi - angle, angle)
    angle = np.where(dy < 0, -angle, angle)
    magnitude = np.hypot(dx, dy)
    hue = (angle % (2 * np.pi)) / (2 * np.pi)
    value = np.clip(magnitude / 0.25, 0, 1)
    value[magnitude < 0.03] = 0
    save('gradient_orientation', hsv_to_rgb(np.dstack([hue, np.ones_like(hue), value])))
    hue_strip = np.tile(np.linspace(0, 1, 720), (40, 1))
    save('orientation_legend', hsv_to_rgb(np.dstack([hue_strip, np.ones_like(hue_strip), np.ones_like(hue_strip)])))
    return {
        'dx_interior_error': float(np.max(np.abs(dx[1:-1, 1:-1] - direct_x[1:-1, 1:-1]))),
        'dy_interior_error': float(np.max(np.abs(dy[1:-1, 1:-1] - direct_y[1:-1, 1:-1]))),
        'full_frame_error': float(max(np.max(np.abs(dx - direct_x)), np.max(np.abs(dy - direct_y))))
    }

def sharpening():
    gaussian = gaussian_filter()
    taj = rgb('taj.jpg')
    blurred = filter_rgb(taj, gaussian)
    high = taj - blurred
    for alpha in (0, 1, 3, 6):
        save(f'taj_alpha_{alpha}', taj + alpha * high)
    matcha = rgb('matcha.jpeg')
    save('matcha_blurred', filter_rgb(matcha, gaussian))

def blend_parts(a, b, mask):
    ga, la = gaus_lap_stack(a)
    gb, lb = gaus_lap_stack(b)
    gm, _ = gaus_lap_stack(mask)
    left = [band * weight[:, :, None] for band, weight in zip(la, gm)]
    right = [band * (1 - weight[:, :, None]) for band, weight in zip(lb, gm)]
    blended = [x + y for x, y in zip(left, right)]
    return ga, gb, la, lb, gm, left, right, blended


def save_breakdown(name, left, right, blended):
    # Figure 3.42 layout: levels 0, 2, 4, followed by the full reconstruction.
    for row, level in enumerate((0, 2, 4)):
        columns = [left[level], right[level], blended[level]]
        limit = max(float(np.max(np.abs(column))) for column in columns)
        for col, data in enumerate(columns):
            save(f'{name}_{chr(97 + row * 3 + col)}', 0.5 + data / (2 * max(limit, 1e-8)))
    for col, data in enumerate((left, right, blended)):
        save(f'{name}_{chr(106 + col)}', np.sum(data, axis=0))


def blending():
    apple, orange = rgb('apple.jpeg'), rgb('orange.jpeg')
    mask = np.zeros(apple.shape[:2], dtype=np.float32)
    mask[:, :mask.shape[1] // 2] = 1
    ga, gb, la, lb, gm, left, right, bands = blend_parts(apple, orange, mask)
    save_breakdown('oraple_process', left, right, bands)
    save('oraple_mask', mask)
    save('oraple_hard_seam', apple * mask[:, :, None] + orange * (1 - mask[:, :, None]))
    for i in range(6):
        save(f'mask_level_{i}', gm[i])
        for name, g, l in [('apple', ga, la), ('orange', gb, lb)]:
            save(f'{name}_gaussian_{i}', g[i])
            save(f'{name}_laplacian_{i}', l[i] if i == 5 else l[i] + 0.5)

    sky, street = rgb('sky1.jpg'), rgb('sky2.JPG')
    h, w = sky.shape[:2]
    center = ((w - 1) / 1.5, (h - 1) / 4)
    matrix = np.array([[1, 0, center[0] - 397], [0, 1, center[1] - 350]], dtype=np.float32)
    moved = cv.warpAffine(sky, matrix, (w, h))
    moon_mask = circular_mask_from_points((h, w), center, (center[0], center[1] - 80))
    _, _, _, _, _, left, right, bands = blend_parts(moved, street, moon_mask)
    save('moon_shifted', moved)
    save('moon_mask', moon_mask)
    save('moon_hard_seam', moved * moon_mask[:, :, None] + street * (1 - moon_mask[:, :, None]))
    save_breakdown('moon_process', left, right, bands)
    moon_result = np.clip(np.sum(bands, axis=0), 0, 1)
    # Second custom nonlinear mask, to also cover the stricter prose instruction.
    coast = rgb('sea1.jpeg')
    center2 = (470, 95)
    matrix2 = np.array([[0.5, 0, center2[0] - 397 * 0.5], [0, 0.5, center2[1] - 350 * 0.5]], dtype=np.float32)
    moved2 = cv.warpAffine(sky, matrix2, (coast.shape[1], coast.shape[0]))
    mask2 = circular_mask_from_points(coast.shape, center2, (470, 55))
    *_, coast_bands = blend_parts(moved2, coast, mask2)
    save('moon_coast_source', moved2)
    save('moon_coast_mask', mask2)
    save('moon_coast_blend', np.sum(coast_bands, axis=0))
    sea_mask = np.zeros(coast.shape[:2], dtype=np.float32)
    sea_mask[:sea_mask.shape[0] // 2] = 1
    save('sea_mask', sea_mask)
    gray_a = cv.cvtColor(apple, cv.COLOR_RGB2GRAY)
    gray_b = cv.cvtColor(orange, cv.COLOR_RGB2GRAY)
    *_, gray_bands = blend_parts(np.repeat(gray_a[:, :, None], 3, axis=2),
                                np.repeat(gray_b[:, :, None], 3, axis=2), mask)
    save('oraple_grayscale', np.sum(gray_bands, axis=0))
    return {
        'apple_reconstruction_error': float(np.max(np.abs(np.sum(la, axis=0) - apple))),
        'orange_reconstruction_error': float(np.max(np.abs(np.sum(lb, axis=0) - orange))),
        'moon_saved_result_error': float(np.max(np.abs(moon_result - np.asarray(Image.open(PROJECT_DIR / 'out' / '2_4_sky.png').convert('RGB')) / 255.)))
    }


if __name__ == '__main__':
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    copy_images()
    metrics = {'convolution': check_convolution(), 'dog': edges_and_orientation()}
    sharpening()
    hybrid_color()
    metrics['stacks'] = blending()
    (ASSET_DIR / 'measurements.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))
