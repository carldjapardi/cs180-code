from part2_3 import gaus_lap_stack
import numpy as np


def circular_mask_from_points(shape, center, edge):
    h, w = shape[:2]
    center_x, center_y = center
    edge_x, edge_y = edge
    radius = np.hypot(edge_x - center_x, edge_y - center_y)
    y, x = np.ogrid[:h, :w]
    return ((x - center_x)**2 + (y - center_y)**2 <= radius**2).astype(np.float32)


def blend_images(im1, im2, mask='vert', circle_center=(0.5, 0.5), circle_points=None):
    levels = 5
    _, lapl_1 = gaus_lap_stack(im1, levels=levels)
    _, lapl_2 = gaus_lap_stack(im2, levels=levels)
    h, w = lapl_1[0].shape[:2]
    mask_image = np.zeros((h, w), dtype=np.float32)
    if mask == 'vert':
        mask_image[:, :w//2] = 1
    elif mask == 'hori':
        mask_image[:h//2, :] = 1
    else:
        if circle_points is not None:
            center, edge = circle_points
            mask_image = circular_mask_from_points((h, w), center, edge)
        else:
            center_x_fraction, center_y_fraction = circle_center
            center = ((w - 1) * center_x_fraction, (h - 1) * center_y_fraction)
            edge = (center[0] + min(h, w) / 4, center[1])
            mask_image = circular_mask_from_points((h, w), center, edge)

    gaus_mask, _ = gaus_lap_stack(mask_image, levels=levels)
    lapl_blend = [
        lapl_1[i]*gaus_mask[i][:, :, None] + lapl_2[i]*(1-gaus_mask[i][:, :, None])
        for i in range(len(lapl_1))
    ]
    collapsed_result = np.sum(lapl_blend, axis=0)
    return np.clip(collapsed_result, 0, 1)
