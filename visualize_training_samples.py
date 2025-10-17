#!/usr/bin/env python3
"""
Script để trực quan hóa dữ liệu training từ dataseries.py
Tạo ảnh PNG cho input data và ground truth data

Tính năng:
    - Phân tích thống kê chi tiết bộ dữ liệu (1000 samples)
    - Tạo báo cáo thống kê trên terminal
    - Tạo ảnh trực quan hóa thống kê (dataset_analysis.png)
    - Tạo ảnh training samples với thông tin chi tiết

Cách sử dụng:
    python visualize_training_samples.py
    python visualize_training_samples.py --n_channels 3 --num_samples 10
    python visualize_training_samples.py --help

Ví dụ:
    - N_CHANNELS=4: 4 ảnh input từ 4 timestamp liên tiếp
    - LEADTIME_CONDITIONING=6: 6 leadtime khác nhau cho ground truth
    - Mỗi sample sẽ có 4 ảnh input + 6 ảnh ground truth
    - Tự động phân tích 1000 samples để tạo báo cáo thống kê
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
import os
import sys
import random
import argparse
from scipy import stats

# Cấu hình - có thể thay đổi các giá trị này
N_CHANNELS = 4
LEADTIME_CONDITIONING = 18
IMG_SIZE = (512, 512)
NUM_SAMPLES = 50
NPZ_FILE = "data/patch013_2025-09-04_2025-10-06_patches_512512_float32.npz"
OUTPUT_DIR = "training_visualizations"

def create_visualization_grid(input_data, ground_truth_data, input_times, gt_times, sample_idx, n_channels, leadtime_conditioning):
    """
    Tạo grid trực quan hóa cho một sample theo cấu trúc dataseries.py
    Input: 4 ảnh (4 channels) + 1 ma trận conditioning time (1 giá trị số)
    Output: 1 ảnh (1 channel)
    """
    # Tạo grid 2x6: 4 ảnh input + 1 conditioning time + 1 ảnh output
    fig, axes = plt.subplots(2, 6, figsize=(24, 10))
    fig.suptitle(f'Sample {sample_idx + 1} - Input vs Ground Truth (dataseries.py format)', fontsize=16, fontweight='bold')
    
    # Input data (4 ảnh channels)
    for i in range(4):
        if i < len(input_data):
            ax = axes[0, i]
            img = input_data[i]
            if img.ndim == 3:
                img = img.squeeze()
            
            # Tính toán thống kê
            img_min = img.min()
            img_max = img.max()
            img_mean = img.mean()
            
            # Sử dụng range thực tế của dữ liệu để hiển thị rõ ràng
            im = ax.imshow(img, cmap='viridis', vmin=img_min, vmax=img_max)
            ax.set_title(f'Input Channel {i+1}\n{input_times[i] if i < len(input_times) else "N/A"}\nMin: {img_min:.3f}, Max: {img_max:.3f}, Mean: {img_mean:.3f}', 
                        fontsize=9)
            ax.axis('off')
            
            # Thêm colorbar với ticks rõ ràng
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks([img_min, img_min + (img_max - img_min) * 0.25, img_min + (img_max - img_min) * 0.5, 
                           img_min + (img_max - img_min) * 0.75, img_max])
            cbar.set_ticklabels([f'{img_min:.3f}', f'{img_min + (img_max - img_min) * 0.25:.3f}', 
                               f'{img_min + (img_max - img_min) * 0.5:.3f}', f'{img_min + (img_max - img_min) * 0.75:.3f}', 
                               f'{img_max:.3f}'])
        else:
            axes[0, i].axis('off')
    
    # Conditioning time matrix (1 giá trị số duy nhất)
    ax = axes[0, 4]
    conditioning_value = input_data[4] if len(input_data) > 4 else 0
    ax.text(0.5, 0.5, f'Leadtime\nConditioning\n\nValue: {conditioning_value:.1f}', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax.set_title('Conditioning Time\n(Leadtime Index)', fontsize=9)
    ax.axis('off')
    
    # Ẩn subplot cuối cùng trong hàng input
    axes[0, 5].axis('off')
    
    # Ground truth data (1 ảnh output)
    ax = axes[1, 0]
    img = ground_truth_data[0] if len(ground_truth_data) > 0 else np.zeros((512, 512))
    if img.ndim == 3:
        img = img.squeeze()
    
    # Tính toán thống kê
    img_min = img.min()
    img_max = img.max()
    img_mean = img.mean()
    
    # Sử dụng range thực tế của dữ liệu để hiển thị rõ ràng
    im = ax.imshow(img, cmap='viridis', vmin=img_min, vmax=img_max)
    ax.set_title(f'Ground Truth\n{gt_times[0] if len(gt_times) > 0 else "N/A"}\nMin: {img_min:.3f}, Max: {img_max:.3f}, Mean: {img_mean:.3f}', 
                fontsize=9)
    ax.axis('off')
    
    # Thêm colorbar với ticks rõ ràng
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([img_min, img_min + (img_max - img_min) * 0.25, img_min + (img_max - img_min) * 0.5, 
                   img_min + (img_max - img_min) * 0.75, img_max])
    cbar.set_ticklabels([f'{img_min:.3f}', f'{img_min + (img_max - img_min) * 0.25:.3f}', 
                       f'{img_min + (img_max - img_min) * 0.5:.3f}', f'{img_min + (img_max - img_min) * 0.75:.3f}', 
                       f'{img_max:.3f}'])
    
    # Ẩn các subplot không sử dụng trong hàng ground truth
    for i in range(1, 6):
        axes[1, i].axis('off')
    
    # Thêm label cho các hàng
    fig.text(0.02, 0.75, f'INPUT DATA\n(4 channels + conditioning)', rotation=90, fontsize=14, fontweight='bold', 
             ha='center', va='center')
    fig.text(0.02, 0.25, f'GROUND TRUTH\n(1 channel)', rotation=90, fontsize=14, fontweight='bold', 
             ha='center', va='center')
    
    plt.tight_layout()
    return fig

def normalize_time_string(ts_str):
    """Normalize timestamp strings to format YYYYMMDDTHHMMSS."""
    s = str(ts_str)
    if "T" in s:
        date_part, time_part = s.split("T", 1)
        date_part = date_part.replace("-", "")
        if len(time_part) == 4:
            time_part = time_part + "00"
        elif len(time_part) == 6:
            pass
        else:
            time_part = (time_part + "000000")[:6]
        return f"{date_part}T{time_part}"
    s = s.replace("-", "")
    if "T" not in s and len(s) >= 8:
        return s
    return s

def analyze_dataset_statistics(all_data, all_times, num_samples=1000):
    """
    Phân tích thống kê chi tiết về bộ dữ liệu
    """
    print("\n" + "="*80)
    print("DATASET STATISTICAL ANALYSIS")
    print("="*80)
    
    # Lấy random samples để phân tích
    if len(all_data) > num_samples:
        sample_indices = random.sample(range(len(all_data)), num_samples)
        sample_data = all_data[sample_indices]
        sample_times = [all_times[i] for i in sample_indices]
    else:
        sample_data = all_data
        sample_times = all_times
        num_samples = len(all_data)
    
    print(f"Analyzing {num_samples} random samples from {len(all_data)} total timesteps...")
    
    # Thống kê cơ bản
    print(f"\nBASIC STATISTICS:")
    print(f"  Total timesteps: {len(all_data)}")
    print(f"  Image shape: {all_data.shape[1:]} (H x W x C)")
    print(f"  Data type: {all_data.dtype}")
    print(f"  Memory usage: {all_data.nbytes / (1024**2):.1f} MB")
    
    # Thống kê giá trị
    print(f"\nVALUE STATISTICS:")
    print(f"  Global min: {all_data.min():.6f}")
    print(f"  Global max: {all_data.max():.6f}")
    print(f"  Global mean: {all_data.mean():.6f}")
    print(f"  Global std: {all_data.std():.6f}")
    print(f"  Global median: {np.median(all_data):.6f}")
    
    # Thống kê non-zero values
    non_zero_mask = all_data > 0
    non_zero_count = np.sum(non_zero_mask)
    non_zero_ratio = non_zero_count / all_data.size
    
    print(f"\nNON-ZERO VALUE STATISTICS:")
    print(f"  Non-zero pixels: {non_zero_count:,} ({non_zero_ratio:.2%})")
    print(f"  Zero pixels: {all_data.size - non_zero_count:,} ({1-non_zero_ratio:.2%})")
    
    if non_zero_count > 0:
        non_zero_values = all_data[non_zero_mask]
        print(f"  Non-zero min: {non_zero_values.min():.6f}")
        print(f"  Non-zero max: {non_zero_values.max():.6f}")
        print(f"  Non-zero mean: {non_zero_values.mean():.6f}")
        print(f"  Non-zero std: {non_zero_values.std():.6f}")
        print(f"  Non-zero median: {np.median(non_zero_values):.6f}")
    
    # Thống kê theo từng ảnh
    print(f"\nPER-IMAGE STATISTICS (from {num_samples} samples):")
    sample_means = []
    sample_stds = []
    sample_maxs = []
    sample_means_nonzero = []
    
    for i in range(min(num_samples, len(sample_data))):
        img = sample_data[i].squeeze()
        sample_means.append(img.mean())
        sample_stds.append(img.std())
        sample_maxs.append(img.max())
        
        # Mean of non-zero values
        nonzero_values = img[img > 0]
        if len(nonzero_values) > 0:
            sample_means_nonzero.append(nonzero_values.mean())
        else:
            sample_means_nonzero.append(0.0)
    
    sample_means = np.array(sample_means)
    sample_stds = np.array(sample_stds)
    sample_maxs = np.array(sample_maxs)
    sample_means_nonzero = np.array(sample_means_nonzero)
    
    print(f"  Mean of image means: {sample_means.mean():.6f} ± {sample_means.std():.6f}")
    print(f"  Mean of image stds: {sample_stds.mean():.6f} ± {sample_stds.std():.6f}")
    print(f"  Mean of image maxs: {sample_maxs.mean():.6f} ± {sample_maxs.std():.6f}")
    print(f"  Mean of non-zero means: {sample_means_nonzero.mean():.6f} ± {sample_means_nonzero.std():.6f}")
    
    # Phân tích temporal
    print(f"\nTEMPORAL ANALYSIS:")
    print(f"  Time range: {all_times[0]} to {all_times[-1]}")
    print(f"  Total duration: {len(all_times)} timesteps")
    
    # Phân tích histogram
    print(f"\nHISTOGRAM ANALYSIS:")
    hist, bin_edges = np.histogram(all_data, bins=50, range=(0, 100))
    print(f"  Histogram bins: {len(hist)}")
    print(f"  Most common value range: {bin_edges[np.argmax(hist)]:.3f} - {bin_edges[np.argmax(hist)+1]:.3f} mm")
    print(f"  Count in most common range: {hist[np.argmax(hist)]:,}")
    
    return {
        'sample_data': sample_data,
        'sample_times': sample_times,
        'sample_means': sample_means,
        'sample_stds': sample_stds,
        'sample_maxs': sample_maxs,
        'sample_means_nonzero': sample_means_nonzero,
        'non_zero_ratio': non_zero_ratio,
        'hist': hist,
        'bin_edges': bin_edges
    }

def create_dataset_visualization(stats_data, output_dir):
    """
    Tạo ảnh trực quan hóa thống kê bộ dữ liệu
    """
    print(f"\nCreating dataset visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dataset Statistical Analysis', fontsize=16, fontweight='bold')
    
    # 1. Histogram của tất cả giá trị
    ax1 = axes[0, 0]
    hist, bin_edges = stats_data['hist'], stats_data['bin_edges']
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax1.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], alpha=0.7, color='skyblue')
    ax1.set_title('Rainfall Distribution (All Data)')
    ax1.set_xlabel('Rainfall (mm)')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram của non-zero values
    ax2 = axes[0, 1]
    sample_data = stats_data['sample_data']
    non_zero_values = sample_data[sample_data > 0]
    if len(non_zero_values) > 0:
        ax2.hist(non_zero_values, bins=50, alpha=0.7, color='lightcoral')
        ax2.set_title('Non-Zero Rainfall Distribution')
        ax2.set_xlabel('Rainfall (mm)')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No non-zero values', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Non-Zero Rainfall Distribution')
    
    # 3. Box plot của mean values per image
    ax3 = axes[0, 2]
    ax3.boxplot([stats_data['sample_means'], stats_data['sample_means_nonzero']], 
                tick_labels=['All Pixels', 'Non-Zero Only'])
    ax3.set_title('Mean Rainfall per Image')
    ax3.set_ylabel('Mean Rainfall (mm)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Scatter plot: Mean vs Std per image
    ax4 = axes[1, 0]
    ax4.scatter(stats_data['sample_means'], stats_data['sample_stds'], alpha=0.6, s=20)
    ax4.set_xlabel('Mean Rainfall per Image (mm)')
    ax4.set_ylabel('Std Rainfall per Image (mm)')
    ax4.set_title('Mean vs Standard Deviation')
    ax4.grid(True, alpha=0.3)
    
    # 5. Time series của mean values
    ax5 = axes[1, 1]
    ax5.plot(stats_data['sample_means'], alpha=0.7, linewidth=1)
    ax5.set_xlabel('Sample Index')
    ax5.set_ylabel('Mean Rainfall (mm)')
    ax5.set_title('Mean Rainfall Over Time')
    ax5.grid(True, alpha=0.3)
    
    # 6. Heatmap của một ảnh mẫu
    ax6 = axes[1, 2]
    sample_img = stats_data['sample_data'][0].squeeze()
    im = ax6.imshow(sample_img, cmap='viridis', aspect='equal', vmin=0, vmax=100)
    ax6.set_title('Sample Rainfall Image (First)')
    ax6.axis('off')
    cbar = plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    cbar.set_label('Rainfall (mm)')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = os.path.join(output_dir, 'dataset_analysis.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")
    return output_path

def load_and_prepare_data(npz_file):
    """
    Load dữ liệu từ file NPZ và chuẩn bị cho visualization
    """
    print(f"Loading data from {npz_file}...")
    
    # Load NPZ file
    data = np.load(npz_file)
    all_data = data['arr_0']  # Shape: (num_timesteps, height, width, channels)
    all_times = data['arr_1']  # Shape: (num_timesteps,)
    
    print(f"Loaded data shape: {all_data.shape}")
    print(f"Loaded times shape: {all_times.shape}")
    print(f"Time range: {all_times[0]} to {all_times[-1]}")
    print(f"Data range: {all_data.min():.3f} to {all_data.max():.3f} mm")
    
    # Không chuẩn hóa - giữ nguyên range [0, 100] mm
    print("Using original rainfall data range [0, 100] mm")
    
    return all_data, all_times

def create_training_samples(all_data, all_times, n_channels, leadtime_conditioning, num_samples):
    """
    Tạo training samples giống như DataSeriesGenerator
    """
    print("Creating training samples...")
    
    # Tạo TOC (Table of Contents) cho dữ liệu
    toc = {}
    for i, t in enumerate(all_times):
        t_normalized = normalize_time_string(t)
        toc[t_normalized] = {"index": i, "time": t_normalized}
    
    # Tạo elements list
    elements = [normalize_time_string(t) for t in all_times]
    elements.sort()
    
    # Tạo placeholder samples giống như trong LazyDataSeries
    placeholder = []
    step = 1  # reuse_y_as_x = False, nên step = 1
    n_fut = leadtime_conditioning  # Số timesteps cần cho ground truth
    
    # Tạo samples với input là n_channels timestamps liên tiếp
    # và ground truth là leadtime_conditioning timestamps tiếp theo
    # Logic giống như trong dataseries.py
    i = 0
    while i <= len(elements) - (n_channels + n_fut):
        # Input timestamps (n_channels)
        input_timestamps = elements[i:i + n_channels]
        
        # Tạo sample cho mỗi leadtime
        for lt_idx in range(leadtime_conditioning):
            x_sample = input_timestamps + [lt_idx]  # Thêm leadtime index
            y_sample = elements[i + n_channels + lt_idx]  # Ground truth timestamp
            placeholder.append([x_sample, y_sample])
        
        i += step
    
    print(f"Total samples created: {len(placeholder)}")
    
    # Lấy random samples
    total_samples = len(placeholder)
    if total_samples < num_samples:
        print(f"Warning: Only {total_samples} samples available, using all")
        sample_indices = list(range(total_samples))
    else:
        sample_indices = random.sample(range(total_samples), num_samples)
    
    samples = []
    for idx in sample_indices:
        sample = placeholder[idx]
        samples.append((idx, sample))
    
    return samples, all_data, all_times, toc

def extract_sample_data(sample, all_data, all_times, toc, n_channels, leadtime_conditioning):
    """
    Extract input và ground truth data từ một sample theo cấu trúc dataseries.py
    Input: 4 ảnh (4 channels) + 1 conditioning time (leadtime index)
    Output: 1 ảnh (1 channel)
    """
    X, Y = sample
    
    # Extract input timestamps (4 channels)
    input_timestamps = X[:n_channels]
    leadtime_idx = X[n_channels]  # Leadtime conditioning index
    
    # Extract ground truth timestamp
    gt_timestamp = Y
    
    # Get actual input data (4 ảnh)
    x_data = []
    x_times = []
    for ts in input_timestamps:
        if ts in toc:
            idx = toc[ts]["index"]
            x_data.append(all_data[idx])
            x_times.append(ts)
        else:
            x_data.append(np.zeros_like(all_data[0]))
            x_times.append("N/A")
    
    # Add conditioning time (leadtime index) as 5th element
    x_data.append(leadtime_idx)
    x_times.append(f"Leadtime {leadtime_idx}")
    
    # Get ground truth data (1 ảnh duy nhất)
    y_data = []
    y_times = []
    
    # Tìm ground truth timestamp tương ứng với leadtime này
    # Logic giống như trong dataseries.py: y = elems[i + self.n_channels + lt]
    elements = sorted(toc.keys())
    
    # Tìm vị trí bắt đầu của input sequence (timestamp đầu tiên)
    start_idx = elements.index(input_timestamps[0]) if input_timestamps[0] in elements else -1
    
    # Tìm timestamp tương ứng với leadtime này
    if start_idx >= 0 and start_idx + n_channels + leadtime_idx < len(elements):
        gt_ts = elements[start_idx + n_channels + leadtime_idx]
        if gt_ts in toc:
            idx = toc[gt_ts]["index"]
            y_data.append(all_data[idx])
            y_times.append(gt_ts)
        else:
            y_data.append(np.zeros_like(all_data[0]))
            y_times.append("N/A")
    else:
        # Nếu không tìm thấy, tạo ảnh trống
        y_data.append(np.zeros_like(all_data[0]))
        y_times.append("N/A")
    
    return x_data, y_data, x_times, y_times

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize training samples from NPZ data')
    parser.add_argument('--npz_file', type=str, default=NPZ_FILE,
                       help=f'Path to NPZ file (default: {NPZ_FILE})')
    parser.add_argument('--n_channels', type=int, default=N_CHANNELS,
                       help=f'Number of input channels (default: {N_CHANNELS})')
    parser.add_argument('--leadtime_conditioning', type=int, default=LEADTIME_CONDITIONING,
                       help=f'Number of leadtime conditioning (default: {LEADTIME_CONDITIONING})')
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES,
                       help=f'Number of samples to visualize (default: {NUM_SAMPLES})')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    return parser.parse_args()

def main():
    """
    Main function để tạo visualizations
    """
    args = parse_arguments()
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"  - SEED: {args.seed}")
    else:
        print(f"  - SEED: None (random)")
    
    print("=" * 60)
    print("TRAINING DATA VISUALIZATION")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - N_CHANNELS: {args.n_channels}")
    print(f"  - LEADTIME_CONDITIONING: {args.leadtime_conditioning}")
    print(f"  - IMG_SIZE: {IMG_SIZE}")
    print(f"  - NUM_SAMPLES: {args.num_samples}")
    print(f"  - NPZ_FILE: {args.npz_file}")
    print(f"  - OUTPUT_DIR: {args.output_dir}")
    print("=" * 60)
    
    # Load data
    all_data, all_times = load_and_prepare_data(args.npz_file)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze dataset statistics
    stats_data = analyze_dataset_statistics(all_data, all_times, num_samples=1000)
    
    # Create dataset visualization
    dataset_viz_path = create_dataset_visualization(stats_data, output_dir)
    
    # Create training samples
    samples, all_data, all_times, toc = create_training_samples(
        all_data, all_times, args.n_channels, args.leadtime_conditioning, args.num_samples)
    
    # Generate visualizations
    print(f"\nGenerating {len(samples)} visualizations...")
    
    for i, (sample_idx, sample) in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)} (index {sample_idx})...")
        
        try:
            # Extract data
            x_data, y_data, x_times, y_times = extract_sample_data(
                sample, all_data, all_times, toc, args.n_channels, args.leadtime_conditioning)
            
            # Print statistics for each image
            print(f"  Input data statistics:")
            for j, img in enumerate(x_data):
                if j < 4:  # 4 ảnh đầu tiên
                    if img.ndim == 3:
                        img = img.squeeze()
                    print(f"    Input Channel {j+1}: Min={img.min():.4f}, Max={img.max():.4f}, Mean={img.mean():.4f}")
                else:  # Conditioning time (giá trị số)
                    print(f"    Conditioning Time: Value={img:.1f}")
            
            print(f"  Ground truth data statistics:")
            for j, img in enumerate(y_data):
                if img.ndim == 3:
                    img = img.squeeze()
                print(f"    GT: Min={img.min():.4f}, Max={img.max():.4f}, Mean={img.mean():.4f}")
            
            # Create visualization
            fig = create_visualization_grid(
                x_data, y_data, x_times, y_times, i, args.n_channels, args.leadtime_conditioning)
            
            # Save figure
            output_path = os.path.join(output_dir, f"sample_{i+1:02d}_index_{sample_idx:04d}.png")
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  Saved: {output_path}")
            
        except Exception as e:
            print(f"  Error processing sample {i+1}: {e}")
            continue
    
    print(f"\nVisualization complete! Check '{output_dir}' directory for results.")
    print(f"Generated files:")
    print(f"  - Dataset analysis: dataset_analysis.png")
    print(f"  - Training samples: {len(samples)} PNG files")
    print("=" * 60)

if __name__ == "__main__":
    main()
