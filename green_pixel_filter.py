#!/usr/bin/env python3
"""
Script to identify and copy images where green pixels (value > 230) comprise >= 5% of the image,
and also copy images that do not meet the criteria to a separate directory.
"""

import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import argparse


def calculate_green_percentage(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            green_channel = img_array[:, :, 1]
            green_high_pixels = np.sum(green_channel > 230)
            total_pixels = green_channel.size
            percentage = (green_high_pixels / total_pixels) * 100
            return percentage
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0.0


def find_and_copy_green_images(source_dir, target_dir, threshold=5.0, recursive=False):
    """
    Find images with green percentage >= threshold and copy to target directory.
    Also copy images that do not meet the criteria to a separate directory.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    nonmatch_target_path = Path(str(target_path) + "_nonmatching")

    target_path.mkdir(parents=True, exist_ok=True)
    nonmatch_target_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}

    if recursive:
        image_files = [f for f in source_path.rglob('*') if f.suffix.lower() in image_extensions]
    else:
        image_files = [f for f in source_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    matching_files = []
    nonmatching_files = []

    print(f"Scanning {len(image_files)} image files...")
    print(f"Looking for images with >= {threshold}% green pixels (value > 230)")
    print("-" * 60)

    for image_file in image_files:
        green_percentage = calculate_green_percentage(image_file)
        print(f"{image_file.name}: {green_percentage:.2f}% green pixels")
        if green_percentage >= threshold:
            matching_files.append((image_file, green_percentage))
            target_file = target_path / image_file.name
            shutil.copy2(image_file, target_file)
            print(f"  ✓ Copied to {target_file}")
        else:
            nonmatching_files.append((image_file, green_percentage))
            nonmatch_file = nonmatch_target_path / image_file.name
            shutil.copy2(image_file, nonmatch_file)
            print(f"  ⨯ Copied to {nonmatch_file}")

    print("-" * 60)
    print(f"Found {len(matching_files)} images matching criteria:")
    for file_path, percentage in matching_files:
        print(f"  - {file_path.name}: {percentage:.2f}%")
    print(f"Found {len(nonmatching_files)} images NOT matching criteria:")
    for file_path, percentage in nonmatching_files:
        print(f"  - {file_path.name}: {percentage:.2f}%")

    if matching_files:
        print(f"\nAll matching files copied to: {target_path}")
    else:
        print(f"\nNo images found with >= {threshold}% green pixels above 230")
    if nonmatching_files:
        print(f"All non-matching files copied to: {nonmatch_target_path}")


def main():
    parser = argparse.ArgumentParser(description='Filter images by green pixel percentage')
    parser.add_argument('source_dir', nargs='?', default='.', 
                       help='Source directory to scan (default: current directory)')
    parser.add_argument('target_dir', nargs='?', default='green_filtered_images',
                       help='Target directory for matching images (default: green_filtered_images)')
    parser.add_argument('--threshold', type=float, default=5.0,
                       help='Minimum percentage of green pixels required (default: 5.0)')
    parser.add_argument('--recursive', action='store_true',
                       help='Search subdirectories recursively')
    
    args = parser.parse_args()
    
    find_and_copy_green_images(
        args.source_dir, 
        args.target_dir, 
        args.threshold, 
        args.recursive
    )


if __name__ == '__main__':
    main()