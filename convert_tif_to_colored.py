#!/usr/bin/env python3
"""
Convert grayscale TIF radar predictions to colored TIF and PNG
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from color_utils import create_cloud_colormap, apply_cloud_colors, visualize_cloud_image
from glob import glob
from tqdm import tqdm
import sys


def parse_command_line():
    parser = argparse.ArgumentParser(description="Convert grayscale TIF to colored TIF and PNG")
    parser.add_argument("--input_dir", action="store", type=str, required=True,
                       help="Directory containing grayscale TIF files")
    parser.add_argument("--output_dir", action="store", type=str, default=None,
                       help="Output directory (default: same as input_dir)")
    parser.add_argument("--pattern", action="store", type=str, default="*.tif",
                       help="File pattern to match (default: *_pred.tif)")
    parser.add_argument("--create_colored_tif", action="store_true", default=False,
                       help="Create colored TIF files (default: True)")
    parser.add_argument("--create_png", action="store_true", default=True,
                       help="Create PNG files (default: True)")
    parser.add_argument("--show_legend", action="store_true", default=True,
                       help="Show legend in PNG files (default: True)")
    parser.add_argument("--dpi", action="store", type=int, default=150,
                       help="DPI for PNG files (default: 150)")
    parser.add_argument("--normalize", action="store_true", default=True,
                       help="Normalize values to cloud range (default: True)")
    
    args = parser.parse_args()
    
    # If output_dir not specified, use input_dir
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    return args


def create_colored_tif(grayscale_tif_path, output_dir, normalize=True):
    """Create colored TIF from grayscale TIF"""
    # Create output directory if it doesn't exist
    colored_dir = os.path.join(output_dir, "colored_tif")
    os.makedirs(colored_dir, exist_ok=True)
    
    # Get filename without extension
    base_filename = os.path.basename(grayscale_tif_path)
    filename_no_ext = os.path.splitext(base_filename)[0]
    
    # Load grayscale image
    img = Image.open(grayscale_tif_path)
    data = np.array(img, dtype=np.float32) / 255.0
    
    # Apply color scheme
    cmap, norm, boundaries = create_cloud_colormap()
    colored_data = apply_cloud_colors(data, cmap, norm, normalize=normalize)
    
    # Convert to uint8 and save
    colored_data_uint8 = (colored_data * 255).astype(np.uint8)
    colored_img = Image.fromarray(colored_data_uint8, mode='RGBA')
    
    # Save colored TIF
    output_filename = f"{filename_no_ext}_colored.tif"
    output_path = os.path.join(colored_dir, output_filename)
    colored_img.save(output_path)
    
    return output_path


def create_png_from_tif(tif_path, output_dir, show_legend=True, dpi=150, normalize=True):
    """Create PNG from TIF file with matplotlib"""
    # Create output directory if it doesn't exist
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
    
    # Get filename without extension
    base_filename = os.path.basename(tif_path)
    filename_no_ext = os.path.splitext(base_filename)[0]
    
    # Extract timestamp from filename (radar_YYYY-MM-DD_HHMM_pred.tif)
    parts = base_filename.split('_')
    if len(parts) >= 3:
        timestamp = f"{parts[1]} {parts[2][:2]}:{parts[2][2:]}"
    else:
        timestamp = filename_no_ext
    
    # Load image
    img = Image.open(tif_path)
    data = np.array(img, dtype=np.float32) / 255.0
    
    # Create visualization
    fig, ax = visualize_cloud_image(
        data,
        title=f"Radar Prediction: {timestamp}",
        show_legend=show_legend,
        normalize=normalize
    )
    
    # Save PNG
    output_filename = f"{filename_no_ext}.png"
    output_path = os.path.join(png_dir, output_filename)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def main():
    args = parse_command_line()
    
    # Find all matching TIF files
    pattern = os.path.join(args.input_dir, args.pattern)
    tif_files = sorted(glob(pattern))
    
    if not tif_files:
        print(f"No files matching pattern '{args.pattern}' found in '{args.input_dir}'")
        sys.exit(1)
    
    print("="*80)
    print("CONVERTING TIF FILES TO COLORED FORMATS")
    print("="*80)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"File pattern:     {args.pattern}")
    print(f"Found {len(tif_files)} matching files")
    print(f"Create colored TIF: {'Yes' if args.create_colored_tif else 'No'}")
    print(f"Create PNG:         {'Yes' if args.create_png else 'No'}")
    print("="*80)
    
    # Track created files
    colored_tifs = []
    pngs = []
    
    # Process each file
    for tif_path in tqdm(tif_files, desc="Converting files"):
        basename = os.path.basename(tif_path)
        
        try:
            # Create colored TIF
            if args.create_colored_tif:
                colored_path = create_colored_tif(
                    tif_path, 
                    args.output_dir,
                    normalize=args.normalize
                )
                colored_tifs.append(colored_path)
            
            # Create PNG directly from grayscale TIF
            if args.create_png:
                png_path = create_png_from_tif(
                    tif_path, 
                    args.output_dir,
                    show_legend=args.show_legend,
                    dpi=args.dpi,
                    normalize=args.normalize
                )
                pngs.append(png_path)
        
        except Exception as e:
            print(f"\nError processing {basename}: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print(f"✅ CONVERSION COMPLETED!")
    if args.create_colored_tif:
        print(f"   Created {len(colored_tifs)} colored TIF files")
        print(f"   Colored TIF directory: {os.path.join(args.output_dir, 'colored_tif')}")
    if args.create_png:
        print(f"   Created {len(pngs)} PNG files")
        print(f"   PNG directory: {os.path.join(args.output_dir, 'png')}")
    print("="*80)


if __name__ == "__main__":
    main()
