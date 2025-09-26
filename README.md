# Pipeline to count GFP+ infected cells 

Pipeline to count cells from GFP fluorescence images. The aim is to separate cells from background, quantify the clean images directly, and, if necessary, to infer cell counts for  unclean (i.e., overexposed) images using an intensity-to-count model fitted based on the clean images.

Clone via

```bash
git clone https://github.com/emkeeler/gfp_cell_counter.git
```

## Files
- `image_threshold.py`: applies histogram-based adaptive thresholding to every image so bright signal stands out from background
- `cell_counter.py`: counts cells in high-quality images by locating blobs after thresholding and logs the results
- `intensity_analyzer.py`: sums green-channel intensity for each image to capture a global brightness metric
- `working_intensity_fit.ipynb`: fits a linear relationship between blob counts and intensities to extrapolate counts for difficult images
- CSV exports under `csv/`: capture intermediate and combined metrics, including `results.csv`, `intensity_processed.csv`, and `results_combined.csv`

## Pipeline
1. **Threshold all images**
   - Run `image_threshold.py` on a directory of  images. The script inspects green-channel histograms and zeroes out values below the strongest occupied bin. 

2. **Select clean images**
   - Manually inspect the thresholded outputs to select clean, well-isolated cells (no overlap, minimal background). 

3. **Count cells in clean images**
   - Execute `cell_counter.py` on the clean image subset. The script parallelizes cells detection, writes per-image comparisons under `comparisons/` for visual QA, and appends counts to `csv/results.csv`.

4. **Measure intensities**
   - Use `intensity_analyzer.py` on both clean and unclean images. The script reports green-channel intensity sums and basic metadata to CSV files such as `csv/intensity_processed.csv` and `csv/intensity_reprocess.csv`.

5. **Fit intensity-to-count model**
   - Open `working_intensity_fit.ipynb`. Load the counts from `csv/results.csv` and corresponding intensities from `csv/intensity_processed.csv`. The script fits a linear regression that maps intensity to observed cell counts for the clean images.

6. **Infer counts for unclean images**
   - Optionally, apply the fitted regression to intensities from the  rejected images. The script combines measured and inferred counts into `csv/results_combined.csv`.

## Requirements
All scripts target Python 3.9+ and rely on the scientific Python stack:

```
pip install numpy pandas matplotlib pillow scikit-image tqdm opencv-python
```

## Usage Notes
- Prefer running thresholding and intensity analysis on the full dataset first so each downstream step has ready inputs.
- Keep `csv/results.csv` under version control or back it up; `cell_counter.py` incrementally appends results while avoiding duplicates.
