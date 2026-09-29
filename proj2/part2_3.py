from pathlib import Path

from filters import blur_2d
import cv2 as cv
import numpy as np


def gaus_lap_stack(im, levels=5):
    if isinstance(im, (str, Path)):
        image = cv.imread(str(im), cv.IMREAD_COLOR)
        im = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32) / 255.
    else:
        im = np.asarray(im, dtype=np.float32)
    gaus_stack = [im]
    for i in range(levels):
        s = 2**i
        h = 2*int(3*s)+1
        current = gaus_stack[-1]
        if current.ndim == 2:
            blurred = blur_2d(current, h=h, s=s)
        else:
            blurred = np.dstack([
                blur_2d(current[:, :, ch], h=h, s=s)
                for ch in range(current.shape[2])
            ])
        gaus_stack.append(blurred)
    lapl_stack = [gaus_stack[i] - gaus_stack[i + 1] for i in range(levels)]
    lapl_stack.append(gaus_stack[-1])
    return gaus_stack, lapl_stack
