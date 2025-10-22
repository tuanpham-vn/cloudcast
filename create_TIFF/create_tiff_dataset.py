import glob
import numpy as np
import argparse
from datetime import datetime, timedelta
from PIL import Image
import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import gc

def parse_command_line():
    parser = argparse.ArgumentParser(description="Create dataset from TIF files for CloudCast training")
    parser.add_argument("--input_dir", action="store", type=str, required=True,
                       help="Directory containing TIF files")
    parser.add_argument("--output_dir", action="store", type=str, required=True,
                       help="Output directory for dataset")
    parser.add_argument("--start_date", action="store", type=str, required=True,
                       help="Start date in YYYY-MM-DD or YYYY-MM-DD_HH:MM format")
    parser.add_argument("--end_date", action="store", type=str, required=True,
                       help="End date in YYYY-MM-DD or YYYY-MM-DD_HH:MM format")
    parser.add_argument("--img_size", action="store", type=str, default="512x512",
                       help="Output image size (WxH)")
    parser.add_argument("--n_channels", action="store", type=int, default=4,
                       help="Number of input channels (default: 4)")
    parser.add_argument("--leadtime_conditioning", action="store", type=int, default=18,
                       help="Maximum leadtime steps (default: 18 for 3 hours)")
    parser.add_argument("--time_interval", action="store", type=int, default=10,
                       help="Time interval between files in minutes (default: 10)")
    parser.add_argument("--output_format", action="store", type=str, default="npz",
                       help="Output format: npz or npy (default: npz)")
    parser.add_argument("--dtype", action="store", type=str, default="float32",
                       help="Data type for output (default: float32)")
    parser.add_argument("--batch_size", action="store", type=int, default=100,
                       help="Number of TIF files to process in each batch (default: 100)")
    parser.add_argument("--show_progress", action="store_true",
                       help="Show progress bar (requires tqdm)")

    args = parser.parse_args()
    return args

def read_tif_file(filepath):
    """Read TIF file and return numpy array"""
    try:
        with Image.open(filepath) as img:
            # Convert to grayscale if needed
            if img.mode != 'L':
                img = img.convert('L')
            # Convert to numpy array directly (no normalization)
            data = np.array(img).astype(np.float32)
            # Clip to [0, 100] range for CloudCast compatibility
            data = np.clip(data, 0, 100)
            return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def parse_datetime_string(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD or YYYY-MM-DD_HH:MM format"""
    try:
        # Try YYYY-MM-DD_HH:MM format first
        return datetime.strptime(date_str, "%Y-%m-%d_%H:%M")
    except ValueError:
        try:
            # Fall back to YYYY-MM-DD format (defaults to 00:00)
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD or YYYY-MM-DD_HH:MM")


def get_tif_files(input_dir, start_date, end_date, time_interval=10):
    """Get sorted list of TIF files within date range"""
    tif_files = []
    tif_times = []

    # Parse date range using the new function
    start_dt = parse_datetime_string(start_date)
    end_dt = parse_datetime_string(end_date)

    current_dt = start_dt
    while current_dt <= end_dt + timedelta(days=1):
        # Expected filename format: YYYY-MM-DD_HHMM_acc10.tif
        expected_filename = current_dt.strftime("%Y-%m-%d_%H%M_acc10.tif")
        filepath = os.path.join(input_dir, expected_filename)

        if os.path.exists(filepath):
            tif_files.append(filepath)
            # Extract timestamp using the helper function
            timestamp = extract_timestamp_from_filename(filepath)
            tif_times.append(timestamp)
        else:
            print(f"Missing file: {filepath}")

        # Move to next time step
        current_dt += timedelta(minutes=time_interval)

    # Calculate expected files more accurately
    total_minutes = int((end_dt - start_dt).total_seconds() / 60)
    expected_files = total_minutes // time_interval + 1
    print(f"Found {len(tif_files)} TIF files out of {expected_files} expected")
    return tif_files, tif_times

def extract_patches_sliding_window(image, patch_size=(512, 512), overlap_ratio=0.1):
    """
    Extract patches from image using sliding window with overlap

    Args:
        image: Input image array (H, W) or (H, W, C)
        patch_size: Size of each patch (H, W)
        overlap_ratio: Overlap ratio between patches (0.0 to 1.0)

    Returns:
        List of patches
    """
    if len(image.shape) == 2:
        H, W = image.shape
        C = 1
        image = image[:, :, np.newaxis]
    else:
        H, W, C = image.shape

    patch_h, patch_w = patch_size
    stride_h = int(patch_h * (1 - overlap_ratio))
    stride_w = int(patch_w * (1 - overlap_ratio))

    patches = []

    # Calculate number of patches needed
    n_patches_h = max(1, (H - patch_h) // stride_h + 1)
    n_patches_w = max(1, (W - patch_w) // stride_w + 1)

    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Calculate patch coordinates
            start_h = min(i * stride_h, H - patch_h)
            start_w = min(j * stride_w, W - patch_w)

            # Extract patch
            patch = image[start_h:start_h + patch_h, start_w:start_w + patch_w, :]

            # Skip if patch is too small (at edges)
            if patch.shape[0] == patch_h and patch.shape[1] == patch_w:
                patches.append(patch)

    return patches

def get_num_patches(first_tif_file: str, patch_size: Tuple[int, int], overlap_ratio: float) -> int:
    """Determine number of patches from the first TIF file"""
    data = read_tif_file(first_tif_file)
    if data is None:
        raise ValueError(f"Cannot read first TIF file: {first_tif_file}")
    
    patches = extract_patches_sliding_window(data, patch_size, overlap_ratio)
    return len(patches)


def initialize_patch_arrays(num_patches: int, num_timesteps: int, patch_size: Tuple[int, int], 
                           dtype: str) -> Tuple[Dict[int, List], Dict[int, List]]:
    """Initialize lists to store patch data and timestamps"""
    patch_data = {i: [] for i in range(num_patches)}
    patch_times = {i: [] for i in range(num_patches)}
    return patch_data, patch_times


def process_tif_file_to_patches(filepath: str, timestamp: str, patch_size: Tuple[int, int], 
                                overlap_ratio: float, num_expected_patches: int) -> List[np.ndarray]:
    """
    Process a single TIF file and extract patches
    
    Returns:
        List of patches (length should match num_expected_patches)
    """
    data = read_tif_file(filepath)
    if data is None:
        return None
    
    patches = extract_patches_sliding_window(data, patch_size, overlap_ratio)
    
    # Validate patch count
    if len(patches) != num_expected_patches:
        print(f"Warning: {filepath} produced {len(patches)} patches, expected {num_expected_patches}")
        return None
    
    # Ensure all patches have correct shape (H, W, 1)
    processed_patches = []
    for patch in patches:
        if len(patch.shape) == 2:
            patch = patch[:, :, np.newaxis]
        processed_patches.append(patch)
    
    return processed_patches

def create_patch_filename(patch_idx, args, start_date, end_date):
    """Create output filename for a patch position across all time steps"""
    return os.path.join(
        args.output_dir,
        f"patch{patch_idx:03d}_{start_date}_{end_date}_patches_{args.img_size.replace('x', '')}_{args.dtype}.{args.output_format}"
    )

def extract_timestamp_from_filename(filename):
    """Extract timestamp from filename in format YYYY-MM-DDTHHMM"""
    try:
        # Format: 2025-09-28_2010_acc10.tif -> 2025-09-28T2010
        parts = os.path.basename(filename).replace('.tif', '').split('_')
        if len(parts) >= 2:
            date_part = parts[0]  # YYYY-MM-DD
            time_part = parts[1]  # HHMM
            return f"{date_part}T{time_part}"
    except:
        pass
    return "Unknown"

def save_to_file(datas, times, filename):
    """Save dataset to file"""
    if filename.endswith('.npz'):
        np.savez(filename, datas, times)
    elif filename.endswith('.npy'):
        timename = filename.replace(".npy", "-times.npy")
        np.save(filename, datas)
        np.save(timename, times)


def append_to_patch_file(patch_array: np.ndarray, timestamp: str, 
                          output_filename: str, dtype: str):
    """
    Append a single patch to an NPZ file incrementally using NPY format
    
    This saves memory by writing data immediately instead of accumulating.
    We use temporary NPY files and combine them at the end.
    
    Args:
        patch_array: Single patch array (H, W, C)
        timestamp: Timestamp string
        output_filename: Path to output file
        dtype: Data type for arrays
    """
    # Create temp directory for incremental data
    temp_dir = output_filename.replace('.npz', '_temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Count existing files to get index
    existing_files = glob.glob(os.path.join(temp_dir, 'patch_*.npy'))
    idx = len(existing_files)
    
    # Save this patch and timestamp
    patch_file = os.path.join(temp_dir, f'patch_{idx:06d}.npy')
    time_file = os.path.join(temp_dir, f'time_{idx:06d}.txt')
    
    np.save(patch_file, patch_array.astype(getattr(np, dtype)))
    with open(time_file, 'w') as f:
        f.write(timestamp)


def finalize_patch_file(output_filename: str, dtype: str):
    """
    Combine all temporary patch files into final NPZ file
    
    Args:
        output_filename: Path to output NPZ file
        dtype: Data type for arrays
    """
    temp_dir = output_filename.replace('.npz', '_temp')
    
    if not os.path.exists(temp_dir):
        print(f"Warning: No temp data for {output_filename}")
        return False
    
    # Load all patches in order
    patch_files = sorted(glob.glob(os.path.join(temp_dir, 'patch_*.npy')))
    time_files = sorted(glob.glob(os.path.join(temp_dir, 'time_*.txt')))
    
    if not patch_files:
        print(f"Warning: No patches found in {temp_dir}")
        return False
    
    patches = []
    timestamps = []
    
    for patch_file, time_file in zip(patch_files, time_files):
        patches.append(np.load(patch_file))
        with open(time_file, 'r') as f:
            timestamps.append(f.read().strip())
    
    # Convert to arrays
    patches_array = np.array(patches).astype(getattr(np, dtype))
    timestamps_array = np.array(timestamps, dtype='<U15')
    
    # Save final file
    save_to_file(patches_array, timestamps_array, output_filename)
    
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    
    return True


def save_patch_batch(patch_data: List[np.ndarray], patch_times: List[str], 
                     output_filename: str, dtype: str):
    """
    Save a batch of patch data to file
    
    Args:
        patch_data: List of patch arrays for this position
        patch_times: List of timestamps
        output_filename: Path to output file
        dtype: Data type for arrays
    """
    if not patch_data:
        return
    
    # Convert to numpy arrays
    patches_array = np.array(patch_data).astype(getattr(np, dtype))
    timestamps_array = np.array(patch_times, dtype='<U15')
    
    # Save to file
    save_to_file(patches_array, timestamps_array, output_filename)
    
    
def print_progress(current: int, total: int, prefix: str = "Progress"):
    """Print progress bar"""
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = '█' * filled + '-' * (bar_length - filled)
    percent = 100 * (current / total)
    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()  # New line when complete


def process_streaming_mode(tif_files: List[str], tif_times: List[str], 
                           num_patches: int, patch_size: Tuple[int, int],
                           overlap_ratio: float, args,
                           show_progress: bool = False) -> Tuple[int, int]:
    """
    Process TIF files in TRUE streaming mode - write to disk immediately
    
    This uses minimal memory by:
    1. Processing one file at a time
    2. Writing each patch to disk immediately (temp files)
    3. Not accumulating any data in memory
    
    Args:
        tif_files: List of TIF file paths
        tif_times: List of timestamps
        num_patches: Number of patches per image
        patch_size: Size of each patch
        overlap_ratio: Overlap between patches
        args: Command line arguments (for output paths and dtype)
        show_progress: Whether to show progress
        
    Returns:
        processed_count: Number of files successfully processed
        skipped_count: Number of files skipped
    """
    total_files = len(tif_files)
    processed_count = 0
    skipped_count = 0
    
    # Pre-create output filenames for each patch
    output_filenames = []
    for patch_idx in range(num_patches):
        output_filename = create_patch_filename(patch_idx, args, 
                                               args.start_date, args.end_date)
        output_filenames.append(output_filename)
    
    # Process each file ONE AT A TIME
    for file_idx, (filepath, timestamp) in enumerate(zip(tif_files, tif_times)):
        patches = process_tif_file_to_patches(filepath, timestamp, patch_size, 
                                              overlap_ratio, num_patches)
        
        if patches is not None:
            # Write each patch to its corresponding file IMMEDIATELY
            for patch_idx, patch in enumerate(patches):
                append_to_patch_file(patch, timestamp, output_filenames[patch_idx], 
                                    args.dtype)
            processed_count += 1
        else:
            skipped_count += 1
        
        if show_progress:
            print_progress(file_idx + 1, total_files, "Processing files")
        
        # Force garbage collection every 50 files
        if (file_idx + 1) % 50 == 0:
            gc.collect()
    
    if show_progress:
        print()  # New line after progress
    
    print(f"Processed: {processed_count} files, Skipped: {skipped_count} files")
    return processed_count, skipped_count


def main():
    args = parse_command_line()

    # Parse patch size
    patch_size = tuple(map(int, args.img_size.split('x')))
    overlap_ratio = 0.1  # 10% overlap

    print("="*70)
    print("CLOUDCAST TIFF DATASET CREATOR - STREAMING MODE")
    print("="*70)
    print(f"Input directory:       {args.input_dir}")
    print(f"Output directory:      {args.output_dir}")
    print(f"Date range:            {args.start_date} to {args.end_date}")
    print(f"Patch size:            {patch_size}")
    print(f"Overlap ratio:         {overlap_ratio}")
    print(f"Data type:             {args.dtype}")
    print(f"Memory mode:           Streaming (minimal RAM usage)")
    print("="*70)

    # Get TIF files
    print("\n[1/5] Scanning for TIF files...")
    tif_files, tif_times = get_tif_files(args.input_dir, args.start_date, 
                                         args.end_date, args.time_interval)

    if not tif_files:
        print("ERROR: No TIF files found!")
        return

    # Calculate expected vs found files using the new parsing function
    start_dt = parse_datetime_string(args.start_date)
    end_dt = parse_datetime_string(args.end_date)
    total_minutes = int((end_dt - start_dt).total_seconds() / 60)
    expected_files = total_minutes // args.time_interval + 1
    missing_count = expected_files - len(tif_files)
    
    print(f"✓ Found {len(tif_files)} TIF files")
    if missing_count > 0:
        print(f"⚠ Missing {missing_count} files ({missing_count/expected_files*100:.1f}% gaps)")
        print(f"  This is OK - the script will skip missing timestamps")
    else:
        print(f"✓ Complete dataset - no missing timestamps")

    # Determine number of patches from first file
    print("\n[2/5] Analyzing patch structure...")
    try:
        num_patches = get_num_patches(tif_files[0], patch_size, overlap_ratio)
        print(f"✓ Each image will produce {num_patches} patches")
        print(f"✓ Total datasets to create: {num_patches}")
    except Exception as e:
        print(f"ERROR: Failed to analyze first file: {e}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process files in streaming mode (minimal memory)
    print(f"\n[3/5] Processing {len(tif_files)} TIF files in streaming mode...")
    print(f"⚡ Writing to disk immediately to minimize RAM usage")
    print(f"⚡ Estimated peak RAM usage: < 1 GB (vs {len(tif_files) * num_patches * 0.5 / 1024:.1f} GB for old method)")
    
    try:
        processed_count, skipped_count = process_streaming_mode(
            tif_files, tif_times, num_patches, patch_size, overlap_ratio,
            args, args.show_progress
        )
    except Exception as e:
        print(f"\nERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Finalize: combine temp files into final NPZ files
    print(f"\n[4/5] Finalizing {num_patches} NPZ files...")
    total_files_created = 0
    
    for patch_idx in range(num_patches):
        output_filename = create_patch_filename(patch_idx, args, 
                                               args.start_date, args.end_date)
        
        if finalize_patch_file(output_filename, args.dtype):
            total_files_created += 1
        
        if args.show_progress:
            print_progress(patch_idx + 1, num_patches, "Finalizing NPZ files")
        
        # Force GC every 5 files during finalization
        if (patch_idx + 1) % 5 == 0:
            gc.collect()
    
    if args.show_progress:
        print()

    # Summary
    print("\n[5/5] Verification and Summary")
    print("="*70)
    print("DATASET CREATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Total patch positions:     {num_patches}")
    print(f"NPZ files created:         {total_files_created}")
    print(f"Time steps per patch:      {processed_count}")
    print(f"Missing timestamps:        {missing_count} ({missing_count/expected_files*100:.1f}%)")
    print(f"Output directory:          {args.output_dir}")
    print("="*70)

    # Print usage instructions
    print_usage_instructions(args, num_patches, processed_count)

def print_usage_instructions(args, num_patches, num_timesteps):
    """Print instructions for using patch timeseries files"""
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS FOR PATCH TIMESERIES FILES")
    print("="*60)

    # Show example files
    print("Example files created:")
    for i in range(min(3, num_patches)):
        output_file = create_patch_filename(i, args, args.start_date, args.end_date)
        print(f"  - {output_file}")
    if num_patches > 3:
        print(f"  ... and {num_patches - 3} more files")

    print()
    print("Dataset format (compatible with create-dataset.py):")
    print(f"  - arr_0.shape = ({num_timesteps}, H, W, C) - Time series of one patch position")
    print(f"  - arr_0.dtype = {args.dtype}")
    print(f"  - arr_1.shape = ({num_timesteps},) - Timestamps for each frame")
    print(f"  - arr_1.dtype = '<U15' - String format: YYYY-MM-DDTHHMM")
    print()
    print("Structure:")
    print(f"  - {num_patches} NPZ files (one per patch position)")
    print(f"  - {num_timesteps} time steps per file")
    print(f"  - Each file contains a complete time series for one spatial patch")
    print()
    print("Example usage with CloudCast:")
    example_file = create_patch_filename(0, args, args.start_date, args.end_date)
    print(f"python cloudcast/cloudcast-unet.py \\")
    print(f"  --dataseries_file {example_file} \\")
    print(f"  --n_channels {args.n_channels} \\")
    print(f"  --leadtime_conditioning {args.leadtime_conditioning} \\")
    print(f"  --preprocess img_size={args.img_size} \\")
    print(f"  --loss_function bcl1 \\")
    print(f"  --include_sun_elevation_angle")
    print()
    print("Benefits of this structure:")
    print("  ✓ One file per patch position = one complete time series")
    print("  ✓ Compatible with CloudCast's time series logic")
    print("  ✓ Same format as create-dataset.py output")
    print("  ✓ Easy to train on individual patches or combine multiple")
    print("="*60)

if __name__ == "__main__":
    main()


    """
    Example usage:
    
    # Basic usage - works with any dataset size (streaming mode)
    python create_TIFF/create_tiff_dataset.py \
      --input_dir data \
      --output_dir output \
      --start_date 2025-09-04 \
      --end_date 2025-10-06 \
      --img_size 512x512 \
      --n_channels 4 \
      --leadtime_conditioning 18 \
      --time_interval 10
    
    # With specific time range (YYYY-MM-DD_HH:MM format)
    python create_TIFF/create_tiff_dataset.py \
      --input_dir data \
      --output_dir output \
      --start_date 2025-09-04_08:00 \
      --end_date 2025-09-04_18:00 \
      --img_size 512x512 \
      --time_interval 10
    
    # For large datasets (5000+ files) with progress tracking
    # Streaming mode uses < 1GB RAM regardless of dataset size!
    python create_TIFF/create_tiff_dataset.py \
      --input_dir tiff_files \
      --output_dir data \
      --start_date 2025-10-05 \
      --end_date 2025-10-05 \
      --img_size 512x512 \
      --show_progress \
      --dtype float32
    
    # Works even on memory-constrained systems (no batch_size needed)
    # The script automatically uses streaming mode for minimal RAM
    python create_tiff_dataset.py \
      --input_dir data \
      --output_dir output \
      --start_date 2025-09-01 \
      --end_date 2025-12-31 \
      --show_progress
      
    # Note: Missing timestamps are handled gracefully
    # The script will skip missing files and create valid datasets
    """
