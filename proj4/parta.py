import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

PROJECT_DIR = Path(__file__).resolve().parent

#A.1
edgeLeft = PROJECT_DIR / 'input/edgeLeft.jpeg'
edgeRight = PROJECT_DIR / 'input/edgeRight.jpeg'
zellerLeft = PROJECT_DIR / 'input/zellerLeft.jpeg'
zellerRight = PROJECT_DIR / 'input/zellerRight.jpeg'
laptopLeft = PROJECT_DIR / 'input/laptopLeft.jpeg'
laptopRight = PROJECT_DIR / 'input/laptopRight.jpeg'

images = [edgeLeft, edgeRight, zellerLeft, zellerRight, laptopLeft, laptopRight]
edgeLeft_im, edgeRight_im, zellerLeft_im, zellerRight_im, laptopLeft_im, laptopRight_im = [plt.imread(i) for i in images]

edgeLeft_points = [[526, 683], [527, 798], [528, 880], [826, 647], [896, 909], [530, 936], [461, 1002]]
edgeRight_points = [[204, 673], [182, 795], [165, 880], [497, 687], [513, 937], [154, 937], [60, 1005]]
zellerLeft_points = [[727, 912], [727, 1052], [884, 1161], [889, 1193], [948, 1017], [539, 910]]
zellerRight_points = [[361, 917], [337, 1051], [458, 1156], [460, 1184], [537, 1021], [169, 902]]
laptopLeft_points = [[626, 121], [800, 326], [715, 796], [900, 816], [840, 951], [529, 1051], [579, 1149]]
laptopRight_points = [[58, 127], [273, 330], [282, 788], [449, 765], [411, 901], [120, 1092], [186, 1172]]

#A.2
def computeH(im1_pts, im2_pts):
    num_corr = len(im1_pts)
    A = np.zeros((num_corr*2, 8))
    b = np.zeros(num_corr*2)
    for i in range(num_corr):
        xi, yi = im1_pts[i]
        xip, yip = im2_pts[i]
        A[2*i] = np.array([xi, yi, 1, 0, 0, 0, -xi*xip, -yi*xip])    
        A[2*i+1] = np.array([0, 0, 0, xi, yi, 1, -xi*yip, -yi*yip])
        b[2*i] = xip
        b[2*i+1] = yip
    h, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return np.append(h, 1).reshape(3, 3)

#A.3
def warpImageNearestNeighbor(im_src, H, output_shape=None):
    h, w, color = im_src.shape
    if output_shape is None:
        corners_src = np.array([[0, 0, 1], [w-1, 0, 1], [0, h-1, 1], [w-1, h-1, 1]]).T
        corners_dest = H @ corners_src
        corners_dest = corners_dest[:2] / corners_dest[2]
        xmin, ymin = np.floor(corners_dest.min(axis=1)).astype(int)
        xmax, ymax = np.ceil(corners_dest.max(axis=1)).astype(int)
        dest_w, dest_h = xmax-xmin+1, ymax-ymin+1
    else:
        dest_h, dest_w = output_shape
        xmin, ymin = 0, 0

    trans_to_origin = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    h_inv = np.linalg.inv(trans_to_origin @ H) # inverse wrapping
    im_dest = np.zeros((dest_h, dest_w, color), dtype=im_src.dtype)
    
    xs = np.arange(dest_w)
    for y in range(im_dest.shape[0]):
        src_coor = h_inv @ np.array([xs, np.full(dest_w, y), np.ones(dest_w)])
        sx, sy, scale = src_coor

        x = np.flatnonzero(~np.isclose(scale, 0))
        sx = np.rint(sx[x] / scale[x]).astype(int)
        sy = np.rint(sy[x] / scale[x]).astype(int)

        valid = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        im_dest[y, x[valid]] = im_src[sy[valid], sx[valid]]
    return im_dest

def warpImageBilinear(im_src, H, output_shape=None):
    h, w, color = im_src.shape
    if output_shape is None:
        corners_src = np.array([[0, 0, 1], [w-1, 0, 1], [0, h-1, 1], [w-1, h-1, 1]]).T
        corners_dest = H @ corners_src
        corners_dest = corners_dest[:2] / corners_dest[2]
        xmin, ymin = np.floor(corners_dest.min(axis=1)).astype(int)
        xmax, ymax = np.ceil(corners_dest.max(axis=1)).astype(int)
        dest_w, dest_h = xmax-xmin+1, ymax-ymin+1
    else:
        dest_h, dest_w = output_shape
        xmin, ymin = 0, 0
    trans_to_origin = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])

    h_inv = np.linalg.inv(trans_to_origin @ H) # inverse wrapping
    im_dest = np.zeros((dest_h, dest_w, color), dtype=float)
    
    xs = np.arange(dest_w)
    for y in range(im_dest.shape[0]):
        src_coor = h_inv @ np.array([xs, np.full(dest_w, y), np.ones(dest_w)])
        sx, sy, scale = src_coor

        x = np.flatnonzero(~np.isclose(scale, 0))
        sx, sy = sx[x] / scale[x], sy[x] / scale[x]

        valid = (sx >= 0) & (sx <= w-1) & (sy >= 0) & (sy <= h-1)
        x, sx, sy = x[valid], sx[valid], sy[valid]

        x_l, y_t = np.floor(sx).astype(int), np.floor(sy).astype(int)
        x_r, y_b = np.minimum(x_l + 1, w-1), np.minimum(y_t + 1, h-1)
        dx, dy = (sx - x_l)[:, None], (sy - y_t)[:, None]
        neigh = [[x_l, y_t, (1-dx)*(1-dy)], [x_l, y_b, (1-dx)*dy], [x_r, y_t, dx*(1-dy)], [x_r, y_b, dx*dy]]
        for nx, ny, weight in neigh:
            im_dest[y, x] += weight * im_src[ny, nx]
    return np.clip(im_dest/255.0, 0, 1)

#Rectification
zeller_rect = [[249, 1174], [459, 1156], [245, 1209], [460, 1187]]
edge_rect = [[41, 524], [148, 601], [6, 631], [124, 700]]
zeller_rect_straight = [[0, 0], [200, 0], [0, 35], [200, 35]]
edge_rect_straight = [[0, 0], [100, 0], [0, 100], [100, 100]]

H_zeller_rect = computeH(zeller_rect, zeller_rect_straight) 
H_edge_rect = computeH(edge_rect, edge_rect_straight) 

zeller_nn = warpImageNearestNeighbor(zellerRight_im, H_zeller_rect, output_shape=(36, 200))
zeller_bil = warpImageBilinear(zellerRight_im, H_zeller_rect, output_shape=(36, 200))
edge_nn = warpImageNearestNeighbor(edgeLeft_im, H_edge_rect, output_shape=(100, 100))
edge_bil = warpImageBilinear(edgeLeft_im, H_edge_rect, output_shape=(100, 100))

plt.imsave(PROJECT_DIR/'out/zeller_rec_bil.jpeg', zeller_bil)
plt.imsave(PROJECT_DIR/'out/zeller_rec_nn.jpeg', zeller_nn)
plt.imsave(PROJECT_DIR/'out/edge_rec_bil.jpeg', edge_bil)
plt.imsave(PROJECT_DIR/'out/edge_rec_nn.jpeg', edge_nn)

#A.4
def blend(warped_im, unwarped_im, warped_mask, unwarped_mask):
    def weight(mask):
        padded = np.pad(mask, 1)
        return distance_transform_edt(padded)[1:-1, 1:-1, None]
    w1, w2 = weight(warped_mask), weight(unwarped_mask)
    total_weight = w1 + w2
    return (w1*warped_im + w2*unwarped_im) / np.where(total_weight > 0, total_weight, 1)

def mosaic(left_points, right_points, left_im, right_im):
    H_mos = computeH(left_points, right_points)
    left_h, left_w, color = left_im.shape
    right_h, right_w = right_im.shape[:2]

    corners_src = np.array([[0, 0, 1], [left_w-1, 0, 1],[0, left_h-1, 1], [left_w-1, left_h-1, 1]]).T
    corners_dest = H_mos @ corners_src
    corners_dest = corners_dest[:2] / corners_dest[2]

    xmin, ymin, xmax, ymax = int(np.floor(min(0, corners_dest[0].min()))), int(np.floor(min(0, corners_dest[1].min()))), int(np.ceil(max(right_w-1, corners_dest[0].max()))), int(np.ceil(max(right_h-1, corners_dest[1].max())))
    dest_h, dest_w = ymax-ymin+1, xmax-xmin+1

    trans_to_origin = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])

    warped_im = warpImageBilinear(left_im, trans_to_origin @ H_mos, output_shape=(dest_h, dest_w))
    unwarped_im = np.zeros_like(warped_im)
    unwarped_im[-ymin:-ymin+right_h, -xmin:-xmin+right_w] = right_im / 255.0

    left_mask = np.ones((left_h, left_w, 1), dtype=np.uint8)
    warped_mask = warpImageNearestNeighbor(left_mask, trans_to_origin @ H_mos, output_shape=(dest_h, dest_w))[:, :, 0].astype(bool)

    unwarped_mask = np.zeros((dest_h, dest_w), dtype=bool)
    unwarped_mask[-ymin:-ymin+right_h, -xmin:-xmin+right_w] = True

    return blend(warped_im, unwarped_im, warped_mask, unwarped_mask)

edge_blended = mosaic(edgeLeft_points, edgeRight_points, edgeLeft_im, edgeRight_im)
zeller_blended = mosaic(zellerLeft_points, zellerRight_points, zellerLeft_im, zellerRight_im)
laptop_blended = mosaic(laptopLeft_points, laptopRight_points, laptopLeft_im, laptopRight_im)

plt.imsave(PROJECT_DIR/'out/mosaic-edge.jpeg', edge_blended)
plt.imsave(PROJECT_DIR/'out/mosaic-zeller.jpeg', zeller_blended)
plt.imsave(PROJECT_DIR/'out/mosaic-laptop.jpeg', laptop_blended)