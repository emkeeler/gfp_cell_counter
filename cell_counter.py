 #!/usr/bin/env python3
"""
Batch blob finder script that processes all images in a directory.
Detects blobs, saves results to a CSV table, and creates side-by-side comparison images.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import blob_log
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_blobs(img_path, min_blob_size=1, max_blob_size=10, threshold=0.1):
    """
    get blobs from an image.
    
    args:
        img_path: path to image file
        min_blob_size: minimum blob size in pixels
        max_blob_size: maximum blob size in pixels
        threshold: threshold for blob detection (lower is more sensitive)
    """
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        arr = img_array
        blobs = blob_log(img_array, min_sigma=min_blob_size, max_sigma=max_blob_size, threshold=threshold)
        return blobs
    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")
        return []

def create_side_by_side_image(img_path, blobs, output_path):
    """
    create side-by-side comparison showing original and blob detection results.
    
    args:
        img_path: path to original image
        blobs: detected blobs
        output_path: path to save comparison image
    """
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        arr = img_array

        fig = plt.Figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.imshow(arr, cmap='gray')
        ax1.axis('off')
        ax2.imshow(arr, cmap='gray')
        if len(blobs) > 0:
            for y, x, r in blobs:
                circle = matplotlib.patches.Circle((x, y), r, color='red', fill=False, linewidth=1)
                ax2.add_patch(circle)
        ax2.set_title(f'Detected Blobs (N = {len(blobs):,})')
        ax2.axis('off')
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        logging.error(f"Error creating comparison image for {img_path}: {e}")

def is_image_file(filename):
    """
    check if a file is an image file.
    
    args:
        filename: name of file to check
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def process_single_image(args):
    """
    process a single image for blob detection.
    
    args:
        args: tuple of (filename, input_dir, comparison_dir, 
                       min_blob_size, max_blob_size, threshold)
    """
    filename, input_dir, comparison_dir, min_blob_size, max_blob_size, threshold = args
    img_path = os.path.join(input_dir, filename)
    blobs = get_blobs(
        img_path, 
        min_blob_size=min_blob_size, 
        max_blob_size=max_blob_size,
        threshold=threshold
    )
    comparison_filename = f"{os.path.splitext(filename)[0]}_comparison.png"
    comparison_path = os.path.join(comparison_dir, comparison_filename)
    create_side_by_side_image(img_path, blobs, comparison_path)
    return {
        'image_name': filename,
        'blob_count': len(blobs),
    }

def process_directory(input_dir, output_dir, contrast_factor=2.0, min_blob_size=1, 
                     max_blob_size=15, threshold=0.1, max_workers=None):
    """
    process all images in a directory for blob detection using threading.
    
    args:
        input_dir: directory containing images to process
        output_dir: directory to save results
        contrast_factor: contrast enhancement factor
        min_blob_size: minimum blob size in pixels
        max_blob_size: maximum blob size in pixels
        threshold: threshold for blob detection
        max_workers: maximum number of worker threads (None for auto)
    """
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)
    comparison_dir = os.path.join(output_dir, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if is_image_file(f)]
    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
        return pd.DataFrame()
    csv_path = os.path.join(output_dir, 'results.csv')
    # read existing csv if present
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            processed_files = set(existing_df['image_name'].astype(str))
        except Exception:
            processed_files = set()
    else:
        processed_files = set()
    # filter out files already in csv
    files_to_process = [f for f in image_files if f not in processed_files]
    if not files_to_process:
        logger.info("All images already processed. Nothing to do.")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df.sort_values('image_name').reset_index(drop=True)
            return df
        else:
            return pd.DataFrame()
    logger.info(f"Found {len(files_to_process)} image files to process")
    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 1) + 4)
    logger.info(f"Using {max_workers} worker threads")
    image_args = [
        (filename, input_dir, comparison_dir, contrast_factor, 
         min_blob_size, max_blob_size, threshold)
        for filename in files_to_process
    ]
    csv_lock = threading.Lock()
    # write header if file does not exist
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            pd.DataFrame([{'image_name': '', 'blob_count': 0}]).iloc[0:0].to_csv(f, index=False)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_filename = {
            executor.submit(process_single_image, args): args[0] 
            for args in image_args
        }
        with tqdm(total=len(files_to_process), desc="Processing images") as pbar:
            for future in as_completed(future_to_filename):
                filename = future_to_filename[future]
                try:
                    result = future.result()
                    logger.info(f"{filename}: {result['blob_count']} blobs detected")
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    result = {'image_name': filename, 'blob_count': 0}
                # write result to csv
                with csv_lock:
                    pd.DataFrame([result]).to_csv(csv_path, mode='a', header=False, index=False)
                pbar.update(1)
    # read all results for return
    df = pd.read_csv(csv_path)
    df = df.sort_values('image_name').reset_index(drop=True)
    logger.info(f"Results saved to {csv_path}")
    logger.info(f"Comparison images saved to {comparison_dir}")
    return df

def main():
    """
    main function to process all images in a directory for blob detection.
    
    args:
        input_dir: directory containing images to process
        output_dir: directory to save results
        min_blob_size: minimum blob size in pixels
        max_blob_size: maximum blob size in pixels
        threshold: threshold for blob detection
        max_workers: maximum number of worker threads (None for auto)
    """
    parser = argparse.ArgumentParser(description='Batch blob detection for multiple images')
    parser.add_argument('input_dir', help='Directory containing images to process')
    parser.add_argument('output_dir', help='Directory to save results')
    parser.add_argument('--min-blob-size', type=int, default=1,
                       help='Minimum blob size in pixels (default: 1)')
    parser.add_argument('--max-blob-size', type=int, default=15,
                       help='Maximum blob size in pixels (default: 15)')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Blob detection threshold - lower is more sensitive (default: 0.1)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of worker threads (default: auto-detect)')
    args = parser.parse_args()
    logger = setup_logging()
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory {args.input_dir} does not exist")
        return 1
    logger.info(f"Processing images in: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Parameters: "
               f"min_blob_size={args.min_blob_size}, max_blob_size={args.max_blob_size}, "
               f"threshold={args.threshold}")
    df = process_directory(
        args.input_dir,
        args.output_dir, 
        contrast_factor=args.contrast_factor,
        min_blob_size=args.min_blob_size,
        max_blob_size=args.max_blob_size,
        threshold=args.threshold,
        max_workers=args.max_workers
    )
    if not df.empty:
        print(df)
    return 0

if __name__ == "__main__":
    exit(main())