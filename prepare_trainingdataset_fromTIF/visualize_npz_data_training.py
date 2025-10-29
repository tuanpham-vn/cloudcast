import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
import os
import argparse

def parse_command_line():
    parser = argparse.ArgumentParser(description="Visualize training data from NPZ file")
    parser.add_argument("--npz_file", action="store", type=str,
                       help="Path to the NPZ file to visualize")
    parser.add_argument("--max_samples", action="store", type=int, default=10,
                       help="Maximum number of samples to display (default: 10)")
    args = parser.parse_args()

    # Use default values if not provided via command line
    if not args.npz_file:
        args.npz_file = "output/tiff_sliding_2025-09-28_2025-09-30_patch512512_overlap10pct_ch4_lt18_float32.npz"  # Default file

    return args

def get_default_args():
    """Get default arguments for debugging in editors"""
    return {
        'npz_file': "/Users/phamtuan/workspace/cloudcast/output/patch000_2025-09-28_2025-09-30_patches_512512_float32.npz",
        'max_samples': 10
    }

def load_npz_data(npz_file):
    """Load data from NPZ file"""
    print(f"Loading data from: {npz_file}")

    if not os.path.exists(npz_file):
        print(f"ERROR: File {npz_file} not found!")
        return None, None

    try:
        # Note: allow_pickle=True is needed because metadata contains dictionaries
        data = np.load(npz_file, allow_pickle=True)
        images = data['arr_0']  # Shape: (N, H, W, C)
        metadata = data['arr_1']  # Metadata for each patch

        print(f"Dataset loaded successfully!")
        print(f"  - Number of samples: {len(images)}")
        print(f"  - Image shape: {images.shape[1:]}")

        return images, metadata

    except Exception as e:
        print(f"ERROR loading NPZ file: {e}")
        print("Note: This script requires allow_pickle=True to load metadata dictionaries.")
        return None, None

def get_sample_info(metadata, sample_idx):
    """Extract information about a specific sample"""
    if isinstance(metadata[sample_idx], dict):
        info = metadata[sample_idx]
        return {
            'original_file': info.get('original_file', 'Unknown'),
            'patch_index': info.get('patch_index', 0),
            'total_patches': info.get('total_patches', 1),
            'timestamp': extract_timestamp_from_filename(info.get('original_file', ''))
        }
    else:
        # Fallback for older format
        return {
            'original_file': f'sample_{sample_idx}',
            'patch_index': 0,
            'total_patches': 1,
            'timestamp': f'T{sample_idx}'
        }

def extract_timestamp_from_filename(filename):
    """Extract timestamp from filename"""
    try:
        # Format: 2024-01-01_0000_acc10.tif
        parts = filename.replace('.tif', '').split('_')
        if len(parts) >= 2:
            date_part = parts[0]
            time_part = parts[1]
            return f"{date_part}T{time_part}"
    except:
        pass
    return "Unknown"

def create_training_sample_visualization(sample_idx, images, metadata, n_channels=4, leadtime_conditioning=18):
    """
    Create visualization for one training sample (sliding window approach)

    Args:
        sample_idx: Index of the sample to visualize
        images: Array of all images (N, H, W, C)
        metadata: Metadata for each image
        n_channels: Number of input channels (for compatibility)
        leadtime_conditioning: Maximum leadtime steps (for compatibility)
    """

    # Get sample info
    sample_info = get_sample_info(metadata, sample_idx)

    # For sliding window approach, each sample is independent
    # Show this patch as a single input
    current_patch = images[sample_idx]

    # Since this is sliding window, we'll show:
    # - Current patch (as input)
    # - A simulated "target" (next patch in sequence, if available)
    # - Or show the same patch twice to demonstrate the concept

    print(f"Sample {sample_idx}: Patch shape {current_patch.shape}")
    print(f"  - Original file: {sample_info['original_file']}")
    print(f"  - Patch index: {sample_info['patch_index'] + 1}/{sample_info['total_patches']}")

    # For demonstration, show the same patch as both input and target
    # In a real scenario, you might want to show adjacent patches
    return create_sliding_window_visualization(current_patch, sample_info, sample_idx)

def create_sliding_window_visualization(current_patch, sample_info, sample_idx):
    """Create visualization for sliding window approach"""

    # Create figure with current patch and a demonstration of overlap
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot current patch
    axes[0].imshow(current_patch.squeeze(), cmap='viridis', vmin=0, vmax=100)
    axes[0].set_title(f'Current Patch\n{sample_info["timestamp"]}', fontsize=12)
    axes[0].axis('off')
    cbar1 = plt.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Rainfall (mm)')

    # For demonstration, show a simulated adjacent patch (with some overlap)
    # In practice, you'd load the actual adjacent patch
    axes[1].imshow(current_patch.squeeze(), cmap='viridis', vmin=0, vmax=100, alpha=0.7)

    # Add rectangle showing overlap region (conceptual)
    rect = patches.Rectangle((current_patch.shape[1] * 0.45, current_patch.shape[0] * 0.45),
                           current_patch.shape[1] * 0.1, current_patch.shape[0] * 0.1,
                           linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    axes[1].add_patch(rect)
    axes[1].text(current_patch.shape[1] * 0.5, current_patch.shape[0] * 0.3,
                'Overlap\nRegion',
                ha='center', va='center', fontsize=10, color='red')

    axes[1].set_title(f'Adjacent Patch (10% Overlap)\n(Demonstration)', fontsize=12)
    axes[1].axis('off')
    cbar2 = plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Rainfall (mm)')

    # Add overall title
    plt.suptitle(f'Sliding Window Sample #{sample_idx + 1} | {sample_info["original_file"]} | Patch {sample_info["patch_index"] + 1}/{sample_info["total_patches"]}',
                 fontsize=14, y=0.98)

    plt.tight_layout()
    return fig

def create_combined_visualization(input_images, target_image, sample_info, sample_idx):
    """Create combined visualization of input and target images (legacy function)"""

    n_inputs = len(input_images)

    # Create figure with subplots
    fig, axes = plt.subplots(1, n_inputs + 1, figsize=(4 * (n_inputs + 1), 4))

    if n_inputs + 1 == 1:
        axes = [axes]

    # Plot input images
    for i, img in enumerate(input_images):
        axes[i].imshow(img.squeeze(), cmap='viridis', vmin=0, vmax=100)
        axes[i].set_title(f'Input {i+1}\n{sample_info["timestamp"]}', fontsize=10)
        axes[i].axis('off')

    # Plot target image
    axes[-1].imshow(target_image.squeeze(), cmap='viridis', vmin=0, vmax=100)
    axes[-1].set_title(f'Target\n(Lead time: +15min)', fontsize=10)
    axes[-1].axis('off')

    # Add overall title
    plt.suptitle(f'Training Sample #{sample_idx + 1} | {sample_info["original_file"]} | Patch {sample_info["patch_index"] + 1}/{sample_info["total_patches"]}',
                 fontsize=12, y=0.98)

    plt.tight_layout()
    return fig

def interactive_visualization(images, metadata, max_samples=10):
    """Interactive visualization allowing user to select samples"""

    print(f"\n{'='*60}")
    print("INTERACTIVE VISUALIZATION")
    print(f"{'='*60}")
    print(f"Dataset contains {len(images)} samples")
    print("Enter sample index to visualize (0 to exit):")
    print(f"Valid range: 0 to {len(images) - 1}")

    while True:
        try:
            user_input = input("\nEnter sample index (or 'q' to quit): ").strip()

            if user_input.lower() in ['q', 'quit', 'exit']:
                break

            sample_idx = int(user_input)

            if 0 <= sample_idx < len(images):
                print(f"Visualizing sample {sample_idx}...")

                fig = create_training_sample_visualization(
                    sample_idx, images, metadata
                )

                if fig:
                    plt.show()
                else:
                    print(f"Could not create visualization for sample {sample_idx}")

            else:
                print(f"Invalid index. Please enter 0 to {len(images) - 1}")

        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nVisualization stopped by user")
            break

def batch_visualization(images, metadata, num_samples=5):
    """Show multiple samples in a grid"""

    print(f"\nShowing {num_samples} random samples...")

    # Select random samples
    total_samples = len(images)
    if total_samples <= num_samples:
        sample_indices = list(range(total_samples))
    else:
        import random
        sample_indices = random.sample(range(total_samples), num_samples)

    # Create subplots
    cols = min(3, num_samples)  # Max 3 columns
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else [[axes]]
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, sample_idx in enumerate(sample_indices):
        row, col = idx // cols, idx % cols

        if sample_idx >= len(images):
            continue

        # Create sample visualization
        fig_sample = create_training_sample_visualization(sample_idx, images, metadata)

        if fig_sample:
            # Copy the content to the subplot
            for i, ax in enumerate(fig_sample.get_axes()):
                if row < len(axes) and col < len(axes[row]):
                    # This is a simplified approach - in practice you'd need more sophisticated copying
                    pass

        # For now, just show a placeholder
        if row < len(axes) and col < len(axes[row]):
            axes[row, col].text(0.5, 0.5, f'Sample\n{sample_idx}',
                              ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].set_title(f'Sample {sample_idx}')

    plt.tight_layout()
    plt.show()

def main(args=None):
    """Main function that can be called with args or use defaults for debugging"""

    # Use provided args or get defaults for debugging
    if args is None:
        args = argparse.Namespace(**get_default_args())
        print("Using default arguments for debugging...")

    # Load data
    images, metadata = load_npz_data(args.npz_file)

    if images is None:
        return

    print(f"\nDataset loaded: {images.shape[0]} samples of shape {images.shape[1:]}")
    print("Note: This dataset uses sliding window approach with 10% overlap")

    # Interactive visualization
    interactive_visualization(images, metadata, args.max_samples)

def print_usage_examples():
    """Print usage examples"""
    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70)
    print("1. Basic visualization:")
    print("   python visualize_npz_data_training.py --npz_file dataset.npz")
    print()
    print("2. With custom max samples:")
    print("   python visualize_npz_data_training.py --npz_file dataset.npz --max_samples 20")
    print()
    print("3. The script will show:")
    print("   - Interactive prompt to enter sample index")
    print("   - Each sample shows: current patch + overlap demonstration")
    print("   - Metadata: original file, patch index, total patches")
    print("="*70)

if __name__ == "__main__":
    # Check if running from command line or editor
    import sys

    # If running from command line with arguments, use argparse
    if len(sys.argv) > 1:
        print_usage_examples()
        args = parse_command_line()
        main(args)
    else:
        # If running from editor (no command line args), use defaults
        print("🔧 Running in DEBUG mode (no command line arguments)")
        print("💡 You can set breakpoints and debug step by step in your editor")
        print("="*70)
        main()  # Will use default arguments
