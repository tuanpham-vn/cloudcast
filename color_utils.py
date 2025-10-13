#!/usr/bin/env python3
"""
Color utilities for CloudCast visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def get_cloud_color_scheme():
    """Get the cloud color scheme as RGBA values"""
    color_list = [
        [0, 0, 0, 0],  # White with 50% transparency (0.0 mm)
        [74, 60, 140, int(255 * 0.35)],  # Purple        (>= 0.2 mm)
        [0, 0, 206, int(255 * 0.75)],  # Deep Blue     (>= 1.0 mm)
        [24, 146, 255, int(255 * 0.85)],  # Light Blue    (>= 3.0 mm)
        [173, 219, 231, int(255 * 0.9)],  # Very Light Blue (>= 5.0 mm)
        [82, 105, 41, 255],  # Dark Green    (>= 7.0 mm)
        [57, 178, 115, 255],  # Green         (>= 10.0 mm)
        [123, 255, 214, 255],  # Aqua         (>= 15.0 mm)
        [123, 255, 0, 255],  # Light Green   (>= 20.0 mm)
        [255, 255, 0, 255],  # Yellow        (>= 30.0 mm)
        [247, 231, 140, 255],  # Light Yellow  (>= 50.0 mm)
        [222, 186, 132, 255],  # Tan          (>= 70.0 mm)
        [255, 166, 0, 255],  # Orange        (>= 100.0 mm)
        [255, 0, 0, 255],  # Red           (>= 150.0 mm)
    ]
    
    # Convert to normalized RGBA values (0-1)
    colors = np.array(color_list) / 255.0
    return colors

def get_cloud_thresholds():
    """Get the cloud value thresholds for each color"""
    thresholds = [0.0, 0.2, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0]
    return thresholds

def create_cloud_colormap():
    """Create a matplotlib colormap for cloud visualization"""
    colors = get_cloud_color_scheme()
    thresholds = get_cloud_thresholds()
    
    # Create a colormap with the specified colors
    cmap = ListedColormap(colors)
    
    # Set the boundaries for the colormap
    boundaries = thresholds + [float('inf')]  # Add infinity for the last color
    norm = plt.Normalize(vmin=0, vmax=150)  # Set the range
    
    return cmap, norm, boundaries

def normalize_to_cloud_range(image, target_min=1.0, target_max=100.0):
    """Normalize image data to cloud range (1-100 mm)"""
    # Get current data range
    current_min = image.min()
    current_max = image.max()
    
    # Avoid division by zero
    if current_max == current_min:
        # If all values are the same, set to minimum cloud value
        normalized = np.full_like(image, target_min)
    else:
        # Normalize to [0, 1] first
        normalized = (image - current_min) / (current_max - current_min)
        # Scale to target range [target_min, target_max]
        normalized = normalized * (target_max - target_min) + target_min
    
    return normalized

def apply_cloud_colors(image, cmap, norm, normalize=True):
    """Apply cloud colors to an image"""
    # Normalize the image to cloud range (1-100 mm) if requested
    if normalize:
        image = normalize_to_cloud_range(image, target_min=1.0, target_max=100.0)
    
    # Apply the colormap
    colored_image = cmap(norm(image))
    return colored_image

def create_cloud_legend():
    """Create a legend for the cloud color scheme"""
    colors = get_cloud_color_scheme()
    thresholds = get_cloud_thresholds()
    
    # Create legend patches
    patches = []
    labels = []
    
    for i, (color, threshold) in enumerate(zip(colors, thresholds)):
        if i == 0:
            label = f"0.0 mm"
        elif i == len(thresholds) - 1:
            label = f">= {threshold} mm"
        else:
            label = f">= {threshold} mm"
        
        # Convert RGBA to RGB for the patch
        rgb_color = color[:3]
        patches.append(mpatches.Patch(color=rgb_color, label=label))
        labels.append(label)
    
    return patches, labels

def visualize_cloud_image(image, title="Cloud Visualization", save_path=None, show_legend=True, normalize=True):
    """Visualize a cloud image with the color scheme"""
    cmap, norm, boundaries = create_cloud_colormap()
    
    # Apply colors with normalization
    colored_image = apply_cloud_colors(image, cmap, norm, normalize=normalize)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the image
    im = ax.imshow(colored_image)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Cloud Value (mm)', fontsize=12)
    
    # Add legend if requested
    if show_legend:
        patches, labels = create_cloud_legend()
        ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved colored visualization: {save_path}")
    
    return fig, ax
