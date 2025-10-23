#!/usr/bin/env python3
"""
Create dataset from TIF files for CloudCast training (ceil-based full coverage)

Key differences vs create_tiff_dataset.py:
- Uses ceil-based grid computation to fully cover image bounds
- Accepts a desired overlap ratio (default 0.10 == 10%)
- Recomputes exact strides so last patch aligns with image edges
- Guarantees full coverage; number of cols/rows increases as needed

Example: For a 1460x887 image with 512x512 patches and desired_overlap=0.10,
this yields 4 columns x 2 rows = 8 patches (full coverage).
"""

import argparse
import glob
import math
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


# ------------------------------ CLI ------------------------------

def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Create dataset from TIF files (ceil-based full coverage tiling)"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing TIF files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for dataset")
    parser.add_argument("--start_date", type=str, required=True, help="Start date in YYYY-MM-DD or YYYY-MM-DD_HH:MM format")
    parser.add_argument("--end_date", type=str, required=True, help="End date in YYYY-MM-DD or YYYY-MM-DD_HH:MM format")
    parser.add_argument("--img_size", type=str, default="512x512", help="Output image size (WxH)")
    parser.add_argument("--n_channels", type=int, default=4, help="Number of input channels (default: 4)")
    parser.add_argument("--leadtime_conditioning", type=int, default=18, help="Maximum leadtime steps (default: 18)")
    parser.add_argument("--time_interval", type=int, default=10, help="Time interval between files in minutes")
    parser.add_argument("--output_format", type=str, default="npz", help="Output format: npz or npy (default: npz)")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type for output (default: float32)")
    parser.add_argument("--desired_overlap_ratio", type=float, default=0.10, help="Desired overlap ratio (0..1)")
    parser.add_argument("--show_progress", action="store_true", help="Show progress bar (requires tqdm)")
    return parser.parse_args()


# ------------------------------ I/O utils ------------------------------

def read_tif_file(filepath: str) -> np.ndarray | None:
    try:
        with Image.open(filepath) as img:
            if img.mode != "L":
                img = img.convert("L")
            data = np.array(img).astype(np.float32)
            data = np.clip(data, 0, 100)
            return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def extract_timestamp_from_filename(filename: str) -> str:
    """Extract timestamp from filename in format YYYY-MM-DDTHHMM."""
    try:
        base = os.path.basename(filename).replace(".tif", "")
        parts = base.split("_")
        if len(parts) >= 2:
            date_part = parts[0]
            time_part = parts[1]
            return f"{date_part}T{time_part}"
    except Exception:
        pass
    return "Unknown"


def save_to_file(datas, times, filename):
    if filename.endswith(".npz"):
        np.savez(filename, datas, times)
    elif filename.endswith(".npy"):
        timename = filename.replace(".npy", "-times.npy")
        np.save(filename, datas)
        np.save(timename, times)


def create_patch_filename(patch_idx, args, start_date, end_date):
    return os.path.join(
        args.output_dir,
        f"patch{patch_idx:03d}_{start_date}_{end_date}_patches_{args.img_size.replace('x', '')}_{args.dtype}.{args.output_format}",
    )


# ------------------------------ Ceil-based tiling ------------------------------

def compute_full_cover_grid(
    image_h: int,
    image_w: int,
    patch_h: int,
    patch_w: int,
    desired_overlap_ratio: float,
) -> Tuple[List[int], List[int], int, int, float, float]:
    """Compute grid positions to fully cover the image using ceil-based logic.

    Returns:
      - y_positions, x_positions
      - num_rows, num_cols
      - effective_overlap_h, effective_overlap_w (0..1)
    """
    desired_stride_h = max(1, int(round(patch_h * (1.0 - desired_overlap_ratio))))
    desired_stride_w = max(1, int(round(patch_w * (1.0 - desired_overlap_ratio))))

    def min_steps(dim: int, patch: int, stride: int) -> int:
        if dim <= patch:
            return 1
        return int(math.ceil((dim - patch) / float(stride))) + 1

    num_rows = min_steps(image_h, patch_h, desired_stride_h)
    num_cols = min_steps(image_w, patch_w, desired_stride_w)

    def recompute_stride(dim: int, patch: int, steps: int) -> float:
        if steps <= 1:
            return 0.0
        return (dim - patch) / float(steps - 1)

    stride_h = recompute_stride(image_h, patch_h, num_rows)
    stride_w = recompute_stride(image_w, patch_w, num_cols)

    def positions(dim: int, patch: int, steps: int, stride: float) -> List[int]:
        if steps == 1:
            return [0]
        vals = [int(round(i * stride)) for i in range(steps)]
        vals[-1] = dim - patch
        vals = [max(0, min(dim - patch, v)) for v in vals]
        for i in range(1, len(vals)):
            if vals[i] < vals[i - 1]:
                vals[i] = vals[i - 1]
        return vals

    y_positions = positions(image_h, patch_h, num_rows, stride_h)
    x_positions = positions(image_w, patch_w, num_cols, stride_w)

    effective_overlap_h = 0.0 if num_rows <= 1 else 1.0 - (stride_h / float(patch_h))
    effective_overlap_w = 0.0 if num_cols <= 1 else 1.0 - (stride_w / float(patch_w))

    return y_positions, x_positions, num_rows, num_cols, effective_overlap_h, effective_overlap_w


def extract_patches_full_cover(
    image: np.ndarray,
    patch_size: Tuple[int, int],
    desired_overlap_ratio: float,
) -> List[np.ndarray]:
    """Extract patches using the ceil-based full-cover grid."""
    if image.ndim == 2:
        H, W = image.shape
        image = image[:, :, np.newaxis]
    else:
        H, W, _ = image.shape

    patch_h, patch_w = patch_size
    ys, xs, _, _, _, _ = compute_full_cover_grid(H, W, patch_h, patch_w, desired_overlap_ratio)

    patches: List[np.ndarray] = []
    for y in ys:
        for x in xs:
            patch = image[y : y + patch_h, x : x + patch_w, :]
            # Ensure patch has correct shape and dimensions
            if patch.shape[0] == patch_h and patch.shape[1] == patch_w and patch.shape[2] == 1:
                patches.append(patch)
            else:
                print(f"WARNING: Skipping malformed patch at ({y}, {x}) with shape {patch.shape}")
    return patches


# ------------------------------ Dataset logic ------------------------------

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


def get_tif_files(input_dir: str, start_date: str, end_date: str, time_interval: int = 10):
    tif_files: List[str] = []
    tif_times: List[str] = []

    start_dt = parse_datetime_string(start_date)
    end_dt = parse_datetime_string(end_date)

    current_dt = start_dt
    while current_dt <= end_dt:
        expected_filename = current_dt.strftime("%Y-%m-%d_%H%M_acc10.tif")
        filepath = os.path.join(input_dir, expected_filename)
        if os.path.exists(filepath):
            tif_files.append(filepath)
            tif_times.append(extract_timestamp_from_filename(filepath))
        current_dt += timedelta(minutes=time_interval)

    return tif_files, tif_times


def get_num_patches(first_tif_file: str, patch_size: Tuple[int, int], desired_overlap_ratio: float) -> int:
    data = read_tif_file(first_tif_file)
    if data is None:
        raise ValueError(f"Cannot read first TIF file: {first_tif_file}")
    patches = extract_patches_full_cover(data, patch_size, desired_overlap_ratio)
    return len(patches)


def save_to_temp_npz_like(patch_array: np.ndarray, timestamp: str, output_filename: str, dtype: str):
    """Append single patch to temp folder; final merge will create the NPZ."""
    temp_dir = output_filename.replace(".npz", "_temp")
    os.makedirs(temp_dir, exist_ok=True)

    existing = glob.glob(os.path.join(temp_dir, "patch_*.npy"))
    idx = len(existing)

    patch_file = os.path.join(temp_dir, f"patch_{idx:06d}.npy")
    time_file = os.path.join(temp_dir, f"time_{idx:06d}.txt")

    # Debug: Check patch shape before saving
    if idx < 3:  # Only debug first few patches
        print(f"Debug: Saving patch {idx} with shape {patch_array.shape}, size {patch_array.size}")

    np.save(patch_file, patch_array.astype(getattr(np, dtype)))
    with open(time_file, "w") as f:
        f.write(timestamp)


def finalize_patch_file(output_filename: str, dtype: str) -> bool:
    temp_dir = output_filename.replace(".npz", "_temp")
    if not os.path.exists(temp_dir):
        print(f"Warning: No temp data for {output_filename}")
        return False

    patch_files = sorted(glob.glob(os.path.join(temp_dir, "patch_*.npy")))
    time_files = sorted(glob.glob(os.path.join(temp_dir, "time_*.txt")))
    if not patch_files:
        print(f"Warning: No patches found in {temp_dir}")
        return False

    # Debug: Check patch shapes before loading
    print(f"Debug: Loading {len(patch_files)} patches from {temp_dir}")
    patches = []
    for i, p in enumerate(patch_files):
        try:
            patch = np.load(p)
            patches.append(patch)
            if i < 3:  # Show first 3 patches for debugging
                print(f"  Patch {i}: shape={patch.shape}, size={patch.size}")
        except Exception as e:
            print(f"ERROR loading patch {p}: {e}")
            return False

    times = []
    for tf in time_files:
        with open(tf, "r") as f:
            times.append(f.read().strip())

    # Debug: Check if all patches have consistent shape
    if patches:
        unique_shapes = list(set(p.shape for p in patches))
        if len(unique_shapes) > 1:
            print(f"ERROR: Inconsistent patch shapes found: {unique_shapes}")
            return False
        print(f"All patches have consistent shape: {patches[0].shape}")

    patches_array = np.array(patches).astype(getattr(np, dtype))
    times_array = np.array(times, dtype="<U15")

    save_to_file(patches_array, times_array, output_filename)
    shutil.rmtree(temp_dir)
    return True


def process_streaming_mode(
    tif_files: List[str],
    tif_times: List[str],
    patch_size: Tuple[int, int],
    desired_overlap_ratio: float,
    args,
) -> Tuple[int, int, int]:
    processed_count = 0

    # Determine number of patches from first file
    num_patches = get_num_patches(tif_files[0], patch_size, desired_overlap_ratio)

    # Pre-create output filenames for each patch position
    output_filenames = [
        create_patch_filename(patch_idx, args, args.start_date, args.end_date)
        for patch_idx in range(num_patches)
    ]

    for filepath, timestamp in zip(tif_files, tif_times):
        data = read_tif_file(filepath)
        if data is None:
            continue
        patches = extract_patches_full_cover(data, patch_size, desired_overlap_ratio)
        if len(patches) != num_patches:
            # Skip inconsistent tilings to preserve timeseries alignment
            continue
        for patch_idx, patch in enumerate(patches):
            save_to_temp_npz_like(patch, timestamp, output_filenames[patch_idx], args.dtype)
        processed_count += 1

    return processed_count, num_patches, len(tif_files) - processed_count


# ------------------------------ Main ------------------------------

def main():
    args = parse_command_line()

    patch_w, patch_h = map(int, args.img_size.split("x"))
    patch_size = (patch_h, patch_w)

    print("=" * 70)
    print("CLOUDCAST TIFF DATASET CREATOR - FULL COVER (CEIL) MODE")
    print("=" * 70)
    print(f"Input directory:       {args.input_dir}")
    print(f"Output directory:      {args.output_dir}")
    print(f"Date range:            {args.start_date} to {args.end_date}")
    print(f"Patch size:            {patch_size[1]}x{patch_size[0]}")
    print(f"Desired overlap:       {args.desired_overlap_ratio * 100:.1f}%")
    print(f"Data type:             {args.dtype}")
    print("=" * 70)

    tif_files, tif_times = get_tif_files(args.input_dir, args.start_date, args.end_date, args.time_interval)
    if not tif_files:
        print("ERROR: No TIF files found!")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[1/3] Processing TIF files with ceil-based full-cover tiling...")
    processed_count, num_patches, skipped = process_streaming_mode(
        tif_files, tif_times, patch_size, args.desired_overlap_ratio, args
    )

    print("\n[2/3] Finalizing NPZ files...")
    total_files_created = 0
    for patch_idx in range(num_patches):
        output_filename = create_patch_filename(patch_idx, args, args.start_date, args.end_date)
        if finalize_patch_file(output_filename, args.dtype):
            total_files_created += 1

    print("\n[3/3] Summary")
    print("=" * 70)
    print("DATASET CREATION COMPLETED")
    print("=" * 70)
    print(f"Total patch positions:     {num_patches}")
    print(f"NPZ files created:         {total_files_created}")
    print(f"Time steps per patch:      {processed_count}")
    print(f"Skipped frames:            {skipped}")
    print(f"Output directory:          {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
