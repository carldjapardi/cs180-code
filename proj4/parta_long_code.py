
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent

#A.1
edgeLeft = PROJECT_DIR / 'input/edgeLeft.jpeg'
edgeRight = PROJECT_DIR / 'input/edgeRight.jpeg'
zellerLeft = PROJECT_DIR / 'input/zellerLeft.jpeg'
zellerRight = PROJECT_DIR / 'input/zellerRight.jpeg'
edgeLeft_points = [[526, 683], [527, 798], [528, 880], [826, 647], [896, 909], [530, 936], [461, 1002]]
edgeRight_points = [[204, 673], [182, 795], [165, 880], [497, 687], [513, 937], [154, 937], [60, 1005]]
zellerLeft_points = [[727, 912], [727, 1052], [741, 1165], [884, 1161], [889, 1193], [948, 1017], [539, 910]]
zellerRight_points = [[361, 917], [337, 1051], [458, 1156], [460, 1184], [379, 1160], [537, 1021], [169, 902]]

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

print(computeH(edgeLeft_points, edgeRight_points))

#A.3
def warpImageNearestNeighbor(im_src, H):
    h, w, color = im_src.shape
    corners_src = np.array([[0, 0, 1], [w-1, 0, 1], [0, h-1, 1], [w-1, h-1, 1]]).T
    corners_dest = H @ corners_src
    corners_dest = corners_dest/corners_dest[2]
    xmin, xmax, ymin, ymax = int(np.floor(corners_dest[0].min())), int(np.ceil(corners_dest[0].max())), int(np.floor(corners_dest[1].min())), int(np.ceil(corners_dest[1].max()))
    dest_w, dest_h = xmax-xmin+1, ymax-ymin+1
    trans_to_origin = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])

    h_inv = np.linalg.inv(trans_to_origin @ H) # inverse wrapping
    im_dest = np.zeros((dest_h, dest_w, color), dtype=im_src.dtype)

    for y in range(im_dest.shape[0]):
        for x in range(im_dest.shape[1]):
            src_coor = h_inv @ np.array([x, y, 1])
            sx, sy, scale = src_coor
            if np.isclose(scale, 0):
                continue
            sx, sy = int(round(sx/scale)), int(round(sy/scale))
            if 0 <= sx < w and 0 <= sy < h:
                im_dest[y, x] = im_src[sy, sx]

    return im_dest

# edgeLeft = plt.imread(edgeLeft)
# H = computeH(edgeLeft_points, edgeRight_points)
# warped = warpImageNearestNeighbor(edgeLeft, H)
# plt.imshow(warped)
# plt.show()


def warpImageBilinear(im_src, H):
    h, w, color = im_src.shape
    corners_src = np.array([[0, 0, 1], [w-1, 0, 1], [0, h-1, 1], [w-1, h-1, 1]]).T
    corners_dest = H @ corners_src
    corners_dest = corners_dest/corners_dest[2]
    xmin, xmax, ymin, ymax = int(np.floor(corners_dest[0].min())), int(np.ceil(corners_dest[0].max())), int(np.floor(corners_dest[1].min())), int(np.ceil(corners_dest[1].max()))
    dest_w, dest_h = xmax-xmin+1, ymax-ymin+1
    trans_to_origin = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])

    h_inv = np.linalg.inv(trans_to_origin @ H) # inverse wrapping
    im_dest = np.zeros((dest_h, dest_w, color), dtype=float)
    xs = np.arange(dest_w)

    for y in range(im_dest.shape[0]):
        for x in range(im_dest.shape[1]):
            src_coor = h_inv @ np.array([x, y, 1])
            sx, sy, scale = src_coor
            if np.isclose(scale, 0):
                continue
            sx, sy = sx/scale, sy/scale
            if not (0 <= sx <= w-1 and 0 <= sy <= h-1):
                continue
            x_l, y_t = int(np.floor(sx)), int(np.floor(sy))
            x_r, y_b = min(x_l+1, w-1), min(y_t+1, h-1)
            dx, dy = sx-x_l, sy-y_t
            neigh = [[x_l, y_t, (1-dx)*(1-dy)], [x_l, y_b, (1-dx)*dy], [x_r, y_t, dx*(1-dy)], [x_r, y_b, dx*dy]]
            for nx, ny, weight in neigh:
                im_dest[y, x] += weight * im_src[ny, nx]
    return np.clip(im_dest/255.0, 0, 1)

edgeLeft = plt.imread(edgeLeft)
H = computeH(edgeLeft_points, edgeRight_points)
warped = warpImageBilinear(edgeLeft, H)
plt.imshow(warped)
plt.show()