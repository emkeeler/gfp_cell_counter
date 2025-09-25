#!/usr/bin/env python3
"""
Image thresholding utility that applies adaptive thresholding to green channel data.
Processes images by analyzing histogram distribution and applying threshold based on bin counts.
"""

try:
    import cv2
    def read_image(img_path):
        """read image using opencv"""
        return cv2.imread(img_path)
    def save_image(path, arr):
        """save image using opencv"""
        cv2.imwrite(path, arr)
except ImportError:
    from PIL import Image
    def read_image(img_path):
        """read image using PIL, convert to numpy array"""
        return np.array(Image.open(img_path))[..., :3]
    def save_image(path, arr):
        """save image using PIL"""
        Image.fromarray(arr.astype(np.uint8)).save(path)

import numpy as np
import os
from glob import glob

def threshold_image(img_path, threshold_count=3e3):
    """
    apply adaptive thresholding to green channel of image based on histogram analysis.
    
    args:
        img_path: path to input image
        threshold_count: minimum count threshold for histogram bins
        
    returns:
        tuple of (original_green, thresholded_green, threshold_value)
    """
    img = read_image(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return None, None, None
    green = img[..., 1].astype(np.float32)
    min_green = np.min(green)
    max_green = np.max(green)
    green_scaled = ((green - min_green) / (max_green - min_green)) * 255
    data = green_scaled.ravel()
    data = data[(data >= 25) & (data <= 240)]
    hist, bin_edges = np.histogram(data, bins=np.arange(0, 257))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = (bin_centers >= 0) & (bin_centers <= 220) & (hist > threshold_count)
    green_thresholded_scaled = green_scaled.copy()
    if np.any(mask):
        max_bin_val = bin_centers[mask].max()
        green_thresholded_scaled[green_thresholded_scaled <= max_bin_val] = 0
        threshold_val = max_bin_val
    else:
        green_thresholded_scaled[:] = 0
        threshold_val = None

    # undo scaling for output
    green_unscaled = green
    green_thresholded = green_thresholded_scaled / 255 * (max_green - min_green) + min_green

    return green_unscaled, green_thresholded, threshold_val

def process_images(input_dir, output_dir, threshold_count=3e3):
    """
    process all PNG images in input directory and save thresholded results.
    
    args:
        input_dir: directory containing input images
        output_dir: directory to save processed images
        threshold_count: histogram bin count threshold
    """
    if not os.path.isdir(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob(os.path.join(input_dir, "*.png")))
    if not image_paths:
        print(f"No images found in directory: {input_dir}")
        return
    print(f"Processing {len(image_paths)} images with threshold count: {threshold_count}")
    for i, img_path in enumerate(image_paths):
        fname = os.path.basename(img_path)
        print(f"Processing {i+1}/{len(image_paths)}: {fname}")
        green, green_thresholded, threshold_val = threshold_image(img_path, threshold_count)
        if green is None:
            continue
        if threshold_val:
            print(f"  Threshold applied: {threshold_val:.1f}")
        else:
            print("  No signal detected above threshold")
        out_path = os.path.join(output_dir, fname)
        save_image(out_path, green_thresholded.astype(np.uint8))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply adaptive thresholding to images in a directory and save results.")
    parser.add_argument("--input-dir", "-i", required=True, help="Input directory containing images")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for thresholded images")
    parser.add_argument("--threshold", "-t", type=float, default=3e3, help="Threshold count for histogram bins (default: 3000)")
    args = parser.parse_args()
    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        threshold_count=args.threshold
    )