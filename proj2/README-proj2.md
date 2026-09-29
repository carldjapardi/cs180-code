# Project 2: Filters and Frequencies

## Files

- `main.py`: inputs and per-part runners. 
- `filters.py`: the two-loop convolution, separable Gaussian blur, and 2D kernel.
- `part1.py`: finite differences and derivative-of-Gaussian comparisons.
- `part2_1.py`: unsharp masking and the blur-then-sharpen experiment.
- `hybrid_image_starter.py`: hybrid filtering and saved Fourier spectra.
- `align_image_code.py`: the supplied interactive alignment helper.
- `part2_3.py`: Gaussian and Laplacian stacks, with no downsampling.
- `part2_4.py`: multiresolution blending with straight and circular masks.
- `report_assets.py`: four-loop convolution, timing checks, and the supplemental
  figures used in the website.
- `input/`, `out/`: source images and existing results.

Dependencies: Python 3, NumPy, SciPy, OpenCV, Matplotlib, scikit-image, and Pillow.
The existing project uses a GUI Matplotlib backend for interactive alignment.

## Run

Run `python3 main.py` from this directory to create the output images from each part.
Select a different runner at the bottom of `main.py` to run another part. 

To regenerate report figures, keep the sibling repository layout:

```text
180/
  180-code/proj2/
  180-portfolio/proj2/
```

Then run `python3 report_assets.py`. It writes to the portfolio's `proj2/assets`,
leaves the original output files in place, and prints measured numerical checks.
Timings use the same 128 × 128 crop for all methods and the median of three runs.
The added color hybrid experiment is a separate controlled apple/orange comparison;
it does not replace any original hybrid output.

The saved original hybrids depend on manually chosen alignment points that were
not stored in the starter pipeline. Reproducing them exactly requires those same
points; the report uses the existing saved results.
