#!/usr/bin/env python3
"""
Script phân tích thống kê dữ liệu mưa từ TIF files
Tính toán các chỉ số quan trọng và phân cấp mưa theo cường độ
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from datetime import datetime, timedelta
import argparse
from collections import defaultdict
# import seaborn as sns  # Optional dependency
from scipy import stats
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Cấu hình phân cấp mưa (theo mm/h hoặc giá trị tương đương)
RAIN_CATEGORIES = {
    'Không mưa': (0, 0),
    'Mưa rất nhỏ': (0.001, 5),
    'Mưa nhỏ': (5, 10),
    'Mưa vừa': (10, 20),
    'Mưa lớn': (20, 40),
    'Mưa rất lớn': (40, 70),
    'Mưa như trút nước': (70, 100)
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze rainfall statistics from TIF files')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing TIF files')
    parser.add_argument('--output_dir', type=str, default='rainfall_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--start_date', type=str, required=True,
                       help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True,
                       help='End date in YYYY-MM-DD format')
    parser.add_argument('--time_interval', type=int, default=10,
                       help='Time interval between files in minutes (default: 10)')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='Number of files to sample for analysis (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Number of files to process in each batch (default: 50)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--show_progress', action='store_true',
                       help='Show progress bar')
    return parser.parse_args()

def read_tif_file(filepath):
    """Read TIF file and return numpy array without normalization"""
    try:
        with Image.open(filepath) as img:
            if img.mode != 'L':
                img = img.convert('L')
            # Convert to numpy array directly (no normalization)
            data = np.array(img).astype(np.float32)
            # Keep original values as they are (should be 0-100 for rainfall data)
            return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def get_tif_files(input_dir, start_date, end_date, time_interval=10):
    """Get sorted list of TIF files within date range"""
    tif_files = []
    tif_times = []
    
    # Parse date range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    current_dt = start_dt
    while current_dt <= end_dt:
        # Expected filename format: YYYY-MM-DD_HHMM_acc10.tif
        expected_filename = current_dt.strftime("%Y-%m-%d_%H%M_acc10.tif")
        filepath = os.path.join(input_dir, expected_filename)
        
        if os.path.exists(filepath):
            tif_files.append(filepath)
            # Extract timestamp
            timestamp = extract_timestamp_from_filename(filepath)
            tif_times.append(timestamp)
        
        # Move to next time step
        current_dt += timedelta(minutes=time_interval)
    
    return tif_files, tif_times

def extract_timestamp_from_filename(filename):
    """Extract timestamp from filename in format YYYY-MM-DDTHHMM"""
    try:
        parts = os.path.basename(filename).replace('.tif', '').split('_')
        if len(parts) >= 2:
            date_part = parts[0]  # YYYY-MM-DD
            time_part = parts[1]  # HHMM
            return f"{date_part}T{time_part}"
    except:
        pass
    return "Unknown"

def categorize_rainfall(value):
    """Categorize rainfall value into predefined categories"""
    for category, (min_val, max_val) in RAIN_CATEGORIES.items():
        if min_val <= value < max_val:
            return category
    # Handle edge case for exactly 100
    if value == 100:
        return 'Mưa như trút nước'
    return 'Không mưa'

def process_single_file(file_info):
    """Process a single TIF file and return statistics"""
    filepath, timestamp, file_idx = file_info
    
    # Read TIF file
    data = read_tif_file(filepath)
    if data is None:
        return None
    
    # Analyze this file
    file_stats = analyze_rainfall_statistics(data, f"file_{file_idx}")
    
    # Add file-specific info
    file_stats['filepath'] = filepath
    file_stats['timestamp'] = timestamp
    file_stats['file_idx'] = file_idx
    file_stats['data'] = data  # Store data for sample image
    
    return file_stats

def process_batch_parallel(file_batch, num_workers):
    """Process a batch of files in parallel"""
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_file, file_batch)
    
    # Filter out None results (failed files)
    return [result for result in results if result is not None]

def print_progress(current, total, prefix="Progress", start_time=None):
    """Print progress bar with time estimation"""
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = '█' * filled + '-' * (bar_length - filled)
    percent = 100 * (current / total)
    
    if start_time is not None:
        elapsed = time.time() - start_time
        if current > 0:
            eta = elapsed * (total - current) / current
            time_str = f" | ETA: {eta:.0f}s"
        else:
            time_str = ""
    else:
        time_str = ""
    
    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total}){time_str}', end='', flush=True)
    if current == total:
        print()  # New line when complete

def analyze_rainfall_statistics(data, sample_name=""):
    """Analyze rainfall statistics for a single image"""
    # Flatten the data
    flat_data = data.flatten()
    
    # Basic statistics
    stats = {
        'total_pixels': len(flat_data),
        'min': flat_data.min(),
        'max': flat_data.max(),
        'mean': flat_data.mean(),
        'std': flat_data.std(),
        'median': np.median(flat_data),
        'q25': np.percentile(flat_data, 25),
        'q75': np.percentile(flat_data, 75),
        'q90': np.percentile(flat_data, 90),
        'q95': np.percentile(flat_data, 95),
        'q99': np.percentile(flat_data, 99)
    }
    
    # Rain statistics (values > 0)
    rain_mask = flat_data > 0
    rain_pixels = flat_data[rain_mask]
    
    if len(rain_pixels) > 0:
        stats.update({
            'rain_pixels': len(rain_pixels),
            'rain_ratio': len(rain_pixels) / len(flat_data),
            'rain_min': rain_pixels.min(),
            'rain_max': rain_pixels.max(),
            'rain_mean': rain_pixels.mean(),
            'rain_std': rain_pixels.std(),
            'rain_median': np.median(rain_pixels),
            'rain_q25': np.percentile(rain_pixels, 25),
            'rain_q75': np.percentile(rain_pixels, 75),
            'rain_q90': np.percentile(rain_pixels, 90),
            'rain_q95': np.percentile(rain_pixels, 95),
            'rain_q99': np.percentile(rain_pixels, 99)
        })
    else:
        stats.update({
            'rain_pixels': 0,
            'rain_ratio': 0.0,
            'rain_min': 0,
            'rain_max': 0,
            'rain_mean': 0,
            'rain_std': 0,
            'rain_median': 0,
            'rain_q25': 0,
            'rain_q75': 0,
            'rain_q90': 0,
            'rain_q95': 0,
            'rain_q99': 0
        })
    
    # Rainfall category distribution
    category_counts = defaultdict(int)
    for value in flat_data:
        category = categorize_rainfall(value)
        category_counts[category] += 1
    
    stats['category_distribution'] = dict(category_counts)
    
    return stats

def print_rainfall_analysis(all_stats, sample_size, total_files):
    """Print comprehensive rainfall analysis to terminal"""
    print("\n" + "="*80)
    print("RAINFALL DATA ANALYSIS REPORT")
    print("="*80)
    print(f"Sample size: {sample_size} files out of {total_files} total files")
    print(f"Analysis period: {all_stats['start_date']} to {all_stats['end_date']}")
    print("="*80)
    
    # Overall statistics
    print("\n📊 OVERALL STATISTICS:")
    print(f"  Total pixels analyzed: {all_stats['total_pixels']:,}")
    print(f"  Data range: {all_stats['global_min']:.3f} - {all_stats['global_max']:.3f}")
    print(f"  Global mean: {all_stats['global_mean']:.3f}")
    print(f"  Global std: {all_stats['global_std']:.3f}")
    print(f"  Global median: {all_stats['global_median']:.3f}")
    
    # Rain statistics
    print(f"\n🌧️  RAIN STATISTICS (values > 0):")
    print(f"  Rain pixels: {all_stats['rain_pixels']:,} ({all_stats['rain_ratio']:.2%})")
    print(f"  No-rain pixels: {all_stats['no_rain_pixels']:,} ({1-all_stats['rain_ratio']:.2%})")
    print(f"  Rain range: {all_stats['rain_min']:.3f} - {all_stats['rain_max']:.3f}")
    print(f"  Rain mean: {all_stats['rain_mean']:.3f}")
    print(f"  Rain std: {all_stats['rain_std']:.3f}")
    print(f"  Rain median: {all_stats['rain_median']:.3f}")
    
    # Percentiles
    print(f"\n📈 PERCENTILES:")
    print(f"  Q25: {all_stats['q25']:.3f}")
    print(f"  Q75: {all_stats['q75']:.3f}")
    print(f"  Q90: {all_stats['q90']:.3f}")
    print(f"  Q95: {all_stats['q95']:.3f}")
    print(f"  Q99: {all_stats['q99']:.3f}")
    
    # Rainfall categories
    print(f"\n🌦️  RAINFALL CATEGORY DISTRIBUTION:")
    total_pixels = all_stats['total_pixels']
    for category, count in all_stats['category_distribution'].items():
        percentage = (count / total_pixels) * 100
        print(f"  {category:20s}: {count:8,} pixels ({percentage:5.2f}%)")
    
    # Temporal analysis
    print(f"\n⏰ TEMPORAL ANALYSIS:")
    print(f"  Files with rain: {all_stats['files_with_rain']} ({all_stats['files_with_rain_ratio']:.2%})")
    print(f"  Files without rain: {all_stats['files_without_rain']} ({1-all_stats['files_with_rain_ratio']:.2%})")
    print(f"  Average rain ratio per file: {all_stats['avg_rain_ratio_per_file']:.3f}")
    print(f"  Max rain ratio in single file: {all_stats['max_rain_ratio']:.3f}")
    print(f"  Min rain ratio in single file: {all_stats['min_rain_ratio']:.3f}")
    
    # Intensity analysis
    print(f"\n⚡ RAIN INTENSITY ANALYSIS:")
    heavy_rain_categories = ['Mưa lớn', 'Mưa rất lớn', 'Mưa như trút nước']
    heavy_rain_pixels = sum(all_stats['category_distribution'].get(cat, 0) for cat in heavy_rain_categories)
    heavy_rain_ratio = heavy_rain_pixels / total_pixels
    print(f"  Heavy rain pixels: {heavy_rain_pixels:,} ({heavy_rain_ratio:.2%})")
    print(f"  Light rain pixels: {all_stats['category_distribution'].get('Mưa rất nhỏ', 0) + all_stats['category_distribution'].get('Mưa nhỏ', 0):,}")
    print(f"  Moderate rain pixels: {all_stats['category_distribution'].get('Mưa vừa', 0):,}")

def create_rainfall_visualizations(all_stats, output_dir):
    """Create comprehensive rainfall visualizations"""
    print(f"\nCreating rainfall visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    # sns.set_palette("husl")  # Optional dependency
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Rainfall Data Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # 1. Value distribution histogram
    ax1 = plt.subplot(3, 4, 1)
    hist, bin_edges = np.histogram(all_stats['all_values'], bins=50, range=(0, 100))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax1.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], alpha=0.7, color='skyblue')
    ax1.set_title('Value Distribution (All Data)')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3)
    
    # 2. Rain-only distribution
    ax2 = plt.subplot(3, 4, 2)
    rain_values = all_stats['all_values'][all_stats['all_values'] > 0]
    if len(rain_values) > 0:
        ax2.hist(rain_values, bins=50, alpha=0.7, color='lightcoral')
        ax2.set_title('Rain Distribution (Values > 0)')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No rain data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Rain Distribution (Values > 0)')
    
    # 3. Rainfall category pie chart
    ax3 = plt.subplot(3, 4, 3)
    categories = list(all_stats['category_distribution'].keys())
    counts = list(all_stats['category_distribution'].values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    # Only show categories with > 0 pixels
    non_zero_categories = [(cat, count) for cat, count in zip(categories, counts) if count > 0]
    if non_zero_categories:
        cats, counts = zip(*non_zero_categories)
        wedges, texts, autotexts = ax3.pie(counts, labels=cats, autopct='%1.1f%%', colors=colors)
        ax3.set_title('Rainfall Category Distribution')
        # Make percentage text smaller
        for autotext in autotexts:
            autotext.set_fontsize(8)
    else:
        ax3.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Rainfall Category Distribution')
    
    # 4. Box plot of rain values
    ax4 = plt.subplot(3, 4, 4)
    if len(rain_values) > 0:
        ax4.boxplot([rain_values], labels=['Rain Values'])
        ax4.set_title('Rain Values Box Plot')
        ax4.set_ylabel('Value')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No rain data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Rain Values Box Plot')
    
    # 5. Rain ratio per file over time
    ax5 = plt.subplot(3, 4, 5)
    ax5.plot(all_stats['rain_ratios_per_file'], alpha=0.7, linewidth=1)
    ax5.set_title('Rain Ratio Over Time')
    ax5.set_xlabel('File Index')
    ax5.set_ylabel('Rain Ratio')
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative distribution
    ax6 = plt.subplot(3, 4, 6)
    sorted_values = np.sort(all_stats['all_values'])
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax6.plot(sorted_values, cumulative, linewidth=2)
    ax6.set_title('Cumulative Distribution')
    ax6.set_xlabel('Value')
    ax6.set_ylabel('Cumulative Probability')
    ax6.grid(True, alpha=0.3)
    
    # 7. Rain intensity heatmap (sample)
    ax7 = plt.subplot(3, 4, 7)
    if 'sample_image' in all_stats:
        im = ax7.imshow(all_stats['sample_image'], cmap='viridis', aspect='equal')
        ax7.set_title('Sample Rain Image')
        ax7.axis('off')
        plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04)
    else:
        ax7.text(0.5, 0.5, 'No sample image', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Sample Rain Image')
    
    # 8. Rain category bar chart
    ax8 = plt.subplot(3, 4, 8)
    non_zero_categories = [(cat, count) for cat, count in all_stats['category_distribution'].items() if count > 0]
    if non_zero_categories:
        cats, counts = zip(*non_zero_categories)
        bars = ax8.bar(range(len(cats)), counts, color=colors[:len(cats)])
        ax8.set_title('Rainfall Categories (Count)')
        ax8.set_xlabel('Category')
        ax8.set_ylabel('Pixel Count')
        ax8.set_xticks(range(len(cats)))
        ax8.set_xticklabels(cats, rotation=45, ha='right')
        ax8.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    # 9. Rain statistics summary
    ax9 = plt.subplot(3, 4, 9)
    ax9.axis('off')
    stats_text = f"""
    RAINFALL STATISTICS SUMMARY
    
    Total Pixels: {all_stats['total_pixels']:,}
    Rain Pixels: {all_stats['rain_pixels']:,} ({all_stats['rain_ratio']:.2%})
    
    Global Range: {all_stats['global_min']:.3f} - {all_stats['global_max']:.3f}
    Rain Range: {all_stats['rain_min']:.3f} - {all_stats['rain_max']:.3f}
    
    Global Mean: {all_stats['global_mean']:.3f}
    Rain Mean: {all_stats['rain_mean']:.3f}
    
    Q95: {all_stats['q95']:.3f}
    Q99: {all_stats['q99']:.3f}
    
    Files with Rain: {all_stats['files_with_rain']} ({all_stats['files_with_rain_ratio']:.2%})
    """
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    # 10. Rain intensity distribution (log scale)
    ax10 = plt.subplot(3, 4, 10)
    if len(rain_values) > 0:
        ax10.hist(rain_values, bins=50, alpha=0.7, color='lightgreen')
        ax10.set_yscale('log')
        ax10.set_title('Rain Intensity (Log Scale)')
        ax10.set_xlabel('Value')
        ax10.set_ylabel('Count (Log)')
        ax10.grid(True, alpha=0.3)
    else:
        ax10.text(0.5, 0.5, 'No rain data', ha='center', va='center', transform=ax10.transAxes)
        ax10.set_title('Rain Intensity (Log Scale)')
    
    # 11. Rain ratio distribution
    ax11 = plt.subplot(3, 4, 11)
    ax11.hist(all_stats['rain_ratios_per_file'], bins=30, alpha=0.7, color='orange')
    ax11.set_title('Rain Ratio Distribution per File')
    ax11.set_xlabel('Rain Ratio')
    ax11.set_ylabel('Number of Files')
    ax11.grid(True, alpha=0.3)
    
    # 12. Rain category percentage
    ax12 = plt.subplot(3, 4, 12)
    if non_zero_categories:
        cats, counts = zip(*non_zero_categories)
        percentages = [count / all_stats['total_pixels'] * 100 for count in counts]
        bars = ax12.bar(range(len(cats)), percentages, color=colors[:len(cats)])
        ax12.set_title('Rainfall Categories (%)')
        ax12.set_xlabel('Category')
        ax12.set_ylabel('Percentage')
        ax12.set_xticks(range(len(cats)))
        ax12.set_xticklabels(cats, rotation=45, ha='right')
        ax12.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax12.text(bar.get_x() + bar.get_width()/2., height,
                     f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = os.path.join(output_dir, 'rainfall_analysis_dashboard.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")
    return output_path

def main():
    """Main function to analyze rainfall data"""
    args = parse_arguments()
    
    print("="*80)
    print("RAINFALL DATA ANALYSIS")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Sample size: {args.sample_size}")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get TIF files
    print("\n[1/4] Scanning for TIF files...")
    tif_files, tif_times = get_tif_files(args.input_dir, args.start_date, 
                                         args.end_date, args.time_interval)
    
    if not tif_files:
        print("ERROR: No TIF files found!")
        return
    
    print(f"✓ Found {len(tif_files)} TIF files")
    
    # Sample files if needed
    if len(tif_files) > args.sample_size:
        import random
        sample_indices = random.sample(range(len(tif_files)), args.sample_size)
        tif_files = [tif_files[i] for i in sample_indices]
        tif_times = [tif_times[i] for i in sample_indices]
        print(f"✓ Sampled {len(tif_files)} files for analysis")
    
    # Analyze files in parallel batches
    print(f"\n[2/4] Analyzing {len(tif_files)} TIF files in parallel...")
    
    # Determine number of workers
    num_workers = args.num_workers or min(cpu_count(), 8)  # Cap at 8 to avoid memory issues
    batch_size = args.batch_size
    
    print(f"  Using {num_workers} parallel workers")
    print(f"  Batch size: {batch_size} files per batch")
    
    # Prepare file info for parallel processing
    file_infos = [(filepath, timestamp, i) for i, (filepath, timestamp) in enumerate(zip(tif_files, tif_times))]
    
    # Process in batches
    all_results = []
    total_batches = (len(file_infos) + batch_size - 1) // batch_size
    start_time = time.time()
    
    for batch_idx in range(0, len(file_infos), batch_size):
        batch_end = min(batch_idx + batch_size, len(file_infos))
        file_batch = file_infos[batch_idx:batch_end]
        
        # Process this batch in parallel
        batch_results = process_batch_parallel(file_batch, num_workers)
        all_results.extend(batch_results)
        
        # Show progress
        if args.show_progress:
            current_batch = (batch_idx // batch_size) + 1
            print_progress(current_batch, total_batches, "Processing batches", start_time)
    
    if args.show_progress:
        print()  # New line after progress
    
    print(f"✓ Processed {len(all_results)} files successfully")
    
    # Aggregate results
    all_stats = {
        'start_date': args.start_date,
        'end_date': args.end_date,
        'all_values': [],
        'rain_ratios_per_file': [],
        'files_with_rain': 0,
        'files_without_rain': 0,
        'sample_image': None
    }
    
    for result in all_results:
        # Store sample image (first one)
        if all_stats['sample_image'] is None:
            all_stats['sample_image'] = result['data']
        
        # Accumulate data
        all_stats['all_values'].extend(result['data'].flatten())
        all_stats['rain_ratios_per_file'].append(result['rain_ratio'])
        
        if result['rain_ratio'] > 0:
            all_stats['files_with_rain'] += 1
        else:
            all_stats['files_without_rain'] += 1
    
    # Convert to numpy arrays for easier computation
    all_stats['all_values'] = np.array(all_stats['all_values'])
    all_stats['rain_ratios_per_file'] = np.array(all_stats['rain_ratios_per_file'])
    
    # Calculate global statistics
    all_stats.update({
        'total_pixels': len(all_stats['all_values']),
        'global_min': all_stats['all_values'].min(),
        'global_max': all_stats['all_values'].max(),
        'global_mean': all_stats['all_values'].mean(),
        'global_std': all_stats['all_values'].std(),
        'global_median': np.median(all_stats['all_values']),
        'q25': np.percentile(all_stats['all_values'], 25),
        'q75': np.percentile(all_stats['all_values'], 75),
        'q90': np.percentile(all_stats['all_values'], 90),
        'q95': np.percentile(all_stats['all_values'], 95),
        'q99': np.percentile(all_stats['all_values'], 99)
    })
    
    # Rain statistics
    rain_mask = all_stats['all_values'] > 0
    rain_values = all_stats['all_values'][rain_mask]
    
    all_stats.update({
        'rain_pixels': len(rain_values),
        'rain_ratio': len(rain_values) / len(all_stats['all_values']),
        'no_rain_pixels': len(all_stats['all_values']) - len(rain_values),
        'rain_min': rain_values.min() if len(rain_values) > 0 else 0,
        'rain_max': rain_values.max() if len(rain_values) > 0 else 0,
        'rain_mean': rain_values.mean() if len(rain_values) > 0 else 0,
        'rain_std': rain_values.std() if len(rain_values) > 0 else 0,
        'rain_median': np.median(rain_values) if len(rain_values) > 0 else 0,
        'rain_q25': np.percentile(rain_values, 25) if len(rain_values) > 0 else 0,
        'rain_q75': np.percentile(rain_values, 75) if len(rain_values) > 0 else 0,
        'rain_q90': np.percentile(rain_values, 90) if len(rain_values) > 0 else 0,
        'rain_q95': np.percentile(rain_values, 95) if len(rain_values) > 0 else 0,
        'rain_q99': np.percentile(rain_values, 99) if len(rain_values) > 0 else 0
    })
    
    # Temporal statistics
    all_stats.update({
        'files_with_rain_ratio': all_stats['files_with_rain'] / len(tif_files),
        'avg_rain_ratio_per_file': all_stats['rain_ratios_per_file'].mean(),
        'max_rain_ratio': all_stats['rain_ratios_per_file'].max(),
        'min_rain_ratio': all_stats['rain_ratios_per_file'].min()
    })
    
    # Calculate category distribution
    category_counts = defaultdict(int)
    for value in all_stats['all_values']:
        category = categorize_rainfall(value)
        category_counts[category] += 1
    all_stats['category_distribution'] = dict(category_counts)
    
    # Print analysis
    print(f"\n[3/4] Printing analysis results...")
    print_rainfall_analysis(all_stats, len(tif_files), len(tif_files))
    
    # Create visualizations
    print(f"\n[4/4] Creating visualizations...")
    viz_path = create_rainfall_visualizations(all_stats, args.output_dir)
    
    # Calculate performance metrics
    total_time = time.time() - start_time
    files_per_second = len(tif_files) / total_time if total_time > 0 else 0
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Performance metrics:")
    print(f"  Total time: {total_time:.1f} seconds")
    print(f"  Files processed: {len(tif_files)}")
    print(f"  Processing speed: {files_per_second:.1f} files/second")
    print(f"  Parallel workers: {num_workers}")
    print(f"  Batch size: {batch_size}")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"Dashboard image: {viz_path}")
    print("="*80)

if __name__ == "__main__":
    main()
