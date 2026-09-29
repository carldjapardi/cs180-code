from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from hybrid_image_starter import create_hybrid_image
from image_io import OUT_DIR
from part1 import (box_filter, diff_x_filter, diff_y_filter, process_image_p11, show_edges, show_edges_gaus, show_edges_gaus_single)
from part2_1 import sharp_to_blur_to_sharp, unsharp_mask_filter
from part2_3 import gaus_lap_stack
from part2_4 import blend_images

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / 'input'

selfie = INPUT_DIR / 'selfie_.png'
cameraman = INPUT_DIR / 'cameraman.png'
taj = INPUT_DIR / 'taj.jpg'
building = INPUT_DIR / 'building.jpeg'
matcha = INPUT_DIR / 'matcha.jpeg'
derek, nutmeg = INPUT_DIR / 'DerekPicture.jpg', INPUT_DIR / 'nutmeg.jpg'
face1, face2 = INPUT_DIR / 'face1.jpg', INPUT_DIR / 'face2.jpg'
house1, house2 = INPUT_DIR / 'house1.jpg', INPUT_DIR / 'house2.jpg'
houseb1, houseb2 = INPUT_DIR / 'houseb1.jpeg', INPUT_DIR / 'houseb2.jpeg'
cr1, cr2 = INPUT_DIR / 'cr1.jpeg', INPUT_DIR / 'cr2.jpeg'
apple, orange = INPUT_DIR / 'apple.jpeg', INPUT_DIR / 'orange.jpeg'
sky1, sky2 = INPUT_DIR / 'sky1.jpg', INPUT_DIR / 'sky2.jpg'
sea2, sea1 = INPUT_DIR / 'sea1.jpeg', INPUT_DIR / 'sea2.jpeg'

#1.1
def run_part_1_1():
    for name, kernel, signed in (
        ('1_1_box', box_filter, False),
        ('1_1_x', diff_x_filter, True),
        ('1_1_y', diff_y_filter, True),
    ):
        result, reference = process_image_p11(selfie, kernel, name, signed=signed)
        print(f'{name} matches SciPy: {np.allclose(result, reference)}')

# #1.2, 1.3
def run_part_1_2_3():
    show_edges(cameraman)
    dx_blur, dy_blur, _, _ = show_edges_gaus(cameraman)
    dx_dog, dy_dog, _, _ = show_edges_gaus_single(cameraman)
    print('DoG Dx matches blurred Dx:', np.allclose(dx_blur[1:-1, 1:-1], dx_dog[1:-1, 1:-1]))
    print('DoG Dy matches blurred Dy:', np.allclose(dy_blur[1:-1, 1:-1], dy_dog[1:-1, 1:-1]))

#2.1
def run_part_2_1():
    unsharp_mask_filter(taj, 'taj')
    unsharp_mask_filter(building, 'building')
    sharp_to_blur_to_sharp(matcha)

#2.2
def run_part_2_2():
    # create_hybrid_image(derek, nutmeg, 'cat', save=False)
    # create_hybrid_image(face1, face2, 'face', sigma1=4, sigma2=3, save=False)
    # create_hybrid_image(house1, house2, 'house', save=False)
    # create_hybrid_image(houseb1, houseb2, 'houseb', sigma1=2, sigma2=2, save=False)
    create_hybrid_image(cr1, cr2, 'cr7', sigma1=0.5, save_process=True)


#2.3
def run_part_2_3():
    for name, path in (('apple', apple), ('orange', orange)):
        gaus_stack, lapl_stack = gaus_lap_stack(path, levels=5)
        fig, axes = plt.subplots(2, len(gaus_stack), figsize=(18, 7))
        for level in range(len(gaus_stack)):
            axes[0, level].imshow(np.clip(gaus_stack[level], 0, 1))
            axes[0, level].set_title(f'Gaussian level {level}')
            if level == len(lapl_stack) - 1:
                lapl_display = lapl_stack[level]
                lapl_title = f'Low frequency residual {level}'
            else:
                lapl_display = lapl_stack[level] + 0.5
                lapl_title = f'Laplacian level {level}'
            axes[1, level].imshow(np.clip(lapl_display, 0, 1))
            axes[1, level].set_title(lapl_title)
            for row in range(2):
                axes[row, level].axis('off')
        fig.tight_layout()
        OUT_DIR.mkdir(exist_ok=True)
        fig.savefig(OUT_DIR / f'2_3_{name}_stacks.png')
        plt.close(fig)

#2.4
def run_part_2_4():
    oraple = blend_images(apple, orange)
    plt.imsave(OUT_DIR / f'2_4_oraple.png', oraple)
    plt.imshow(oraple)
    plt.show()

    sea_blended = blend_images(sea1, sea2, mask='hori')
    plt.imsave(OUT_DIR / f'2_4_sea.png', sea_blended)
    plt.imshow(sea_blended)
    plt.show()

    sky1_rgb = cv.cvtColor(cv.imread(str(sky1)), cv.COLOR_BGR2RGB).astype(np.float32) / 255.
    h, w = sky1_rgb.shape[:2]
    source_center, source_edge = (397, 350), (397, 270)
    target_center = ((w - 1) / 1.5, (h - 1) / 4)
    shift_x = target_center[0] - source_center[0]
    shift_y = target_center[1] - source_center[1]
    moon_transform = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
    sky1_shifted = cv.warpAffine(sky1_rgb, moon_transform, (w, h))
    target_edge = (source_edge[0] + shift_x, source_edge[1] + shift_y)
    sky_blended = blend_images(sky1_shifted, sky2, mask='circ', circle_points=(target_center, target_edge))
    plt.imsave(OUT_DIR / '2_4_sky.png', sky_blended)
    plt.imshow(sky_blended)
    plt.show()

run_part_2_2()