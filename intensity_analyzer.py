#!/usr/bin/env python3
"""
Script to calculate normalized intensity for multiple images using the robust normalization method
and save results to a CSV file.
"""

import os
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from skimage import exposure
import argparse
from datetime import datetime

def process_image(image_path):
    """
    Process a single image and return its normalized intensity sum.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (filename, normalized_intensity_sum, width, height, channels, error_msg)
    """
    try:
        with Image.open(image_path) as img:
            # convert to numpy array
            
            img_array = np.array(img)
            # get image dimensions
            height, width = img_array.shape[:2]
            channels = img_array.shape[2] if len(img_array.shape) > 2 else 1
            
            if channels == 3:
                img_array = img_array[..., 1]
            else:
                gray_array = img_array
            
            # calculate normalized intensity sum
            #intensity_sum = _robust_normalize_intensity_sum(gray_array)
            intensity_sum = np.sum(gray_array)
            
            return (
                Path(image_path).name,
                float(intensity_sum),
                width,
                height,
                channels,
                None
            )
            
    except Exception as e:
        return (
            Path(image_path).name,
            None,
            None,
            None,
            None,
            str(e)
        )


def analyze_images(source_dir, output_csv, recursive=False):
    """
    Analyze all images in a directory and save results to CSV.
    
    Args:
        source_dir (str): Directory to search for images
        output_csv (str): Output CSV file path
        recursive (bool): Whether to search subdirectories
    """
    source_path = Path(source_dir)
    
    # supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}
    
    # find all image files
    if recursive:
        image_files = [f for f in source_path.rglob('*') if f.suffix.lower() in image_extensions]
    else:
        image_files = [f for f in source_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} image files to process...")
    print("Processing images and calculating normalized intensity...")
    print("-" * 60)
    
    results = []
    
    for i, image_file in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {image_file.name}")
        
        result = process_image(image_file)
        results.append(result)
        
        if result[5] is not None:  # error occurred
            print(f"  ❌ Error: {result[5]}")
        else:
            print(f"  ✓ Intensity sum: {result[1]:.2f}")
    
    # write results to CSV
    print(f"\nWriting results to {output_csv}...")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # write header
        writer.writerow([
            'filename',
            'normalized_intensity_sum',
            'width',
            'height',
            'channels',
            'error'
        ])
        
        # write data rows
        for result in results:
            writer.writerow(result)
    
    # print summary
    successful = [r for r in results if r[5] is None]
    failed = [r for r in results if r[5] is not None]
    
    print("-" * 60)
    print(f"Summary:")
    print(f"  Total images processed: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed files:")
        for result in failed:
            print(f"  - {result[0]}: {result[5]}")
    
    print(f"\nResults saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description='Analyze image intensity using robust normalization')
    parser.add_argument('source_dir', nargs='?', default='.', 
                       help='Source directory to scan (default: current directory)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output CSV file (default: intensity_results_TIMESTAMP.csv)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Search subdirectories recursively')
    
    args = parser.parse_args()
    
    # generate default output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"intensity_results_{timestamp}.csv"
    
    analyze_images(args.source_dir, args.output, args.recursive)


if __name__ == '__main__':
    main()