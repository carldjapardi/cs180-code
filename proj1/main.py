import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def l2_norm(ch1, ch2): # 2d array, one color
    return np.sqrt(np.sum((ch1-ch2)**2))

def normalize_channel(ch):
    return (ch - ch.min()) / (ch.max() - ch.min())

def ncc(ch1, ch2):
    a = (ch1-np.mean(ch1)) 
    b = (ch2-np.mean(ch2)) 
    return np.sum(a*b) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))

def edge_ncc(ch1, ch2):
    gy1, gx1 = np.gradient(ch1)
    gy2, gx2 = np.gradient(ch2)
    edge1 = np.hypot(gx1, gy1)
    edge2 = np.hypot(gx2, gy2)
    return ncc(edge1, edge2)

def plt_show(im):
    plt.figure(figsize=(8, 8))
    plt.imshow(im)       # im_out is RGB: [r, g, b]
    plt.axis("off")
    plt.show()

def basic_align(ch_anchor, ch, ax, edge):
    window = 15
    best_match = -np.inf
    best_d = 0
    for d in range(-1*window, window):
        h, w = ch.shape
        crop_y, crop_x = int(h*0.1), int(w*0.1)
        interior = (slice(crop_y, h-crop_y), slice(crop_x, w-crop_x))
        ch_shifted = np.roll(ch, d, axis=ax)[interior]
        match = ncc(ch_anchor[interior], ch_shifted) if not edge else edge_ncc(ch_anchor[interior], ch_shifted)
        if match > best_match:
            best_match = match
            best_d = d
    aligned = np.roll(ch, best_d, axis=ax)
    return aligned, best_d, best_match
    
def process_image(file_path, filename, num, method=basic_align, edge=False):
    im = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    im = im.astype(np.float32) / 255
    height = int(np.floor(im.shape[0]/3.0)) 
    b, g, r = im[:height, :], im[height:height*2, :], im[2*height:3*height, :]

    ar, r_dy, _ = method(b, r, 0, edge)
    ar, r_dx, r_score = method(b, ar, 1, edge)
    ag, g_dy, _ = method(b, g, 0, edge)
    ag, g_dx, g_score = method(b, ag, 1, edge)

    return_info = True
    if return_info:
        print(f"======= {filename}_{num} ======= ")
        print(f"Red: (dx, dy)=({r_dx}, {r_dy}), r_NCC={r_score:.4f}")
        print(f"Green: (dx, dy)=({g_dx}, {g_dy}), g_NCC={g_score:.4f}")

    im_out = np.dstack([ar, ag, b])
    plt_show(im_out)
    out_uint8 = np.clip(im_out * 255.0, 0, 255).astype(np.uint8)
    out_bgr = cv.cvtColor(out_uint8, cv.COLOR_RGB2BGR)
    if edge:
        cv.imwrite(f'proj1/out/output_edge_{num}_{filename}.jpg', out_bgr)
    else:
        cv.imwrite(f'proj1/out/output_{num}_{filename}.jpg', out_bgr)

def anti_aliasing(ch): #2d array, 1 color channel
    return gaussian_filter(ch, sigma = 1)

def downsampling(ch, k):
    while k > 1:
        ch = anti_aliasing(ch)[::2, ::2]
        k //= 2
    return ch

def pyramid_align(ch_anchor, ch, ax, edge):
    ch_proc = ch
    scale_to_window = {16:15, 8:10, 4:8, 2:5, 1:3}
    total_delta = 0
    for k, window in scale_to_window.items():
        ch_downs = downsampling(ch_proc, k)
        ch_anchor_downs = downsampling(ch_anchor, k)
        h, w = ch_downs.shape
        crop_y, crop_x = int(h * 0.1), int(w * 0.1)
        interior = (slice(crop_y, h - crop_y), slice(crop_x, w - crop_x))
        ch_anchor_cropped = ch_anchor_downs[interior]
        best_match = -np.inf
        best_d = 0
        for d in range(-window, window + 1):
            ch_cropped = np.roll(ch_downs, d, axis=ax)[interior]
            match = ncc(ch_anchor_cropped, ch_cropped) if not edge else edge_ncc(ch_anchor_cropped, ch_cropped)
            if match > best_match:
                best_match = match
                best_d = d
        ch_proc = np.roll(ch_proc, best_d*k, axis=ax)
        total_delta += best_d*k 
        # print(f"scale={k}, residual={best_d}, total_delta={total_delta}, NCC={best_match:.4f}")
    aligned = np.roll(ch, total_delta, axis=ax)
    return aligned, total_delta, best_match

jpg = ['tobolsk', 'cathedral', 'monastery']
for i in range(len(jpg)):
    process_image(f'proj1/input/{jpg[i]}.jpg', jpg[i], i, method=basic_align, edge=True)

extra = ['ex1', 'ex2', 'ex3']
for i in range(len(extra)):
    process_image(f'proj1/input/{extra[i]}.jpg', extra[i], i, method=pyramid_align, edge=False)

tif = ['emir', 'self_portrait', 'melons', 'church', 'harvesters', 'icon', 'ilemselga', 'religous_painting', 'siren', 'three_generations', 'wharf']
for i in range(len(tif)):
    process_image(f'proj1/input/{tif[i]}.tif', tif[i], i, method=pyramid_align, edge=True)


    
