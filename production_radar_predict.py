#!/usr/bin/env python3
"""
Production Radar Prediction Script
Optimized for CPU inference with 16 cores and 120GB RAM

USAGE EXAMPLES:

1. Basic usage with default settings:
   python production_radar_predict.py

2. Specify end timestamp:
   python production_radar_predict.py "2025-10-27-08-40"

3. Specify end timestamp and output directory:
   python production_radar_predict.py "2025-10-27-08-40" "my_predictions"

4. Programmatic usage:
   from production_radar_predict import main
   main("2025-10-27-08-40", "output_dir")

CONFIGURATION:
- Modify CONFIG dictionary to change model path, channels, formats, etc.
- Supports different timestamp formats via 'timestamp_format'
- Handles insufficient images with configurable padding strategies:
  * 'repeat_last': Repeat the last available image
  * 'zero_pad': Fill with zero images
  * 'interpolate': Linear interpolation between first and last images
  * 'auto': Automatically select strategy based on number of missing images

PADDING STRATEGIES:
When fewer than n_channels images are available, the script will:
1. Search for images going backwards from end_timestamp
2. Apply the configured padding strategy to reach n_channels
3. Continue with normal prediction pipeline

Available strategies:
- 'repeat_last': Repeat the last available image
- 'zero_pad': Fill with zero images
- 'interpolate': Linear interpolation between first and last images
- 'auto': Automatically select strategy based on number of missing images
  * If missing images <= auto_padding_threshold (default: 3): Use 'interpolate'
  * If missing images > auto_padding_threshold: Use 'repeat_last'

REQUIREMENTS:
- TensorFlow model weights file
- Input TIF images in the specified directory
- GDAL (optional, for geospatial metadata preservation)
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Dict
from PIL import Image
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# GDAL imports with fallback
try:
    from osgeo import gdal, osr
    GDAL_AVAILABLE = True
except ImportError:
    print("Warning: GDAL not available. Geospatial metadata will not be preserved.")
    GDAL_AVAILABLE = False

# Add workspace to path for model imports
workspace_path = os.path.dirname(os.path.abspath(__file__))
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

import tensorflow as tf
from base.preprocess import create_squeezed_leadtime_conditioning

# Configuration
CONFIG = {
    'model_path': 'model_2025_10_26_1839_flip-180-90.weights.h5',
    'patch_size': 512,
    'n_channels': 9,
    'leadtime_conditioning': 10,
    'batch_size': 8,  # High batch size for CPU with 120GB RAM
    'crop_margin': 0,
    'leadtimes': [i for i in range(17, -1, -1)],  # Customizable leadtime list
    'max_workers': 6,  # Half of CPU cores for parallel processing
    'timestamp_format': '%Y-%m-%d-%H-%M',  # Default timestamp format
    'tif_extension': '.tif',
    'data_directory': 'production_output',  # Default directory to search for TIF files
    'time_interval_minutes': 10,  # Time interval between images in minutes
    'padding_strategy': 'auto',  # Options: 'repeat_last', 'zero_pad', 'interpolate', 'auto'
    'auto_padding_threshold': 3  # If missing images <= this threshold, use interpolate; otherwise use repeat_last
}


def read_geospatial_info(filepath: str) -> Dict:
    """Read geospatial information from TIF file using GDAL"""
    if not GDAL_AVAILABLE:
        return {}
    
    try:
        dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
        if dataset is None:
            return {}
        
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        
        geospatial_info = {
            'geotransform': geotransform,
            'projection': projection,
            'width': dataset.RasterXSize,
            'height': dataset.RasterYSize,
        }
        
        dataset = None
        return geospatial_info
        
    except Exception:
        return {}


def adjust_geotransform_for_crop(geotransform: Tuple, crop_margin: int) -> Tuple:
    """Adjust geotransform for cropped image"""
    if not geotransform:
        return geotransform
    
    # Adjust origin coordinates for crop
    new_geotransform = list(geotransform)
    new_geotransform[0] += crop_margin * geotransform[1]  # Adjust X origin
    new_geotransform[3] += crop_margin * geotransform[5]  # Adjust Y origin
    
    return tuple(new_geotransform)


def save_tif_with_geospatial_info(data: np.ndarray, filepath: str, geospatial_info: Dict):
    """Save numpy array as TIF file with geospatial information"""
    if not GDAL_AVAILABLE or not geospatial_info:
        img = Image.fromarray(data.astype(np.float32), mode='F')
        img.save(filepath)
        return
    
    try:
        driver = gdal.GetDriverByName('GTiff')
        height, width = data.shape
        
        dataset = driver.Create(filepath, width, height, 1, gdal.GDT_Float32)
        
        if 'geotransform' in geospatial_info:
            dataset.SetGeoTransform(geospatial_info['geotransform'])
        
        if 'projection' in geospatial_info:
            dataset.SetProjection(geospatial_info['projection'])
        
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        band.FlushCache()
        
        dataset = None
        
    except Exception:
        img = Image.fromarray(data.astype(np.float32), mode='F')
        img.save(filepath)


def auto_extract_tif_paths(end_timestamp: str, n_channels: int = None, 
                          data_dir: str = None, timestamp_format: str = None) -> List[str]:
    """Automatically extract TIF file paths based on end timestamp"""
    from datetime import datetime, timedelta
    import glob
    
    if n_channels is None:
        n_channels = CONFIG['n_channels']
    if data_dir is None:
        data_dir = CONFIG['data_directory']
    if timestamp_format is None:
        timestamp_format = CONFIG['timestamp_format']
    
    print(f"Auto-extracting {n_channels} TIF files ending at {end_timestamp}")
    
    try:
        # Parse end timestamp
        end_dt = datetime.strptime(end_timestamp, timestamp_format)
    except ValueError as e:
        raise ValueError(f"Cannot parse timestamp '{end_timestamp}' with format '{timestamp_format}': {e}")
    
    # Generate timestamps going backwards
    tif_paths = []
    interval = timedelta(minutes=CONFIG['time_interval_minutes'])
    
    for i in range(n_channels):
        # Calculate timestamp for this image (going backwards)
        img_dt = end_dt - (interval * (n_channels - 1 - i))
        img_timestamp = img_dt.strftime(timestamp_format)
        
        # Try different possible paths
        possible_paths = [
            os.path.join(data_dir, f"{img_timestamp}{CONFIG['tif_extension']}"),
            f"{img_timestamp}{CONFIG['tif_extension']}",  # Current directory
            os.path.join(workspace_path, data_dir, f"{img_timestamp}{CONFIG['tif_extension']}"),
        ]
        
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                tif_paths.append(path)
                found = True
                break
        
        if not found:
            print(f"Warning: Could not find TIF file for timestamp {img_timestamp}")
            print(f"Searched paths: {possible_paths}")
    
    if len(tif_paths) == 0:
        raise FileNotFoundError(f"No TIF files found for sequence ending at {end_timestamp}")
    
    print(f"✓ Found {len(tif_paths)} TIF files out of {n_channels} requested")
    for i, path in enumerate(tif_paths):
        print(f"  [{i+1}] {os.path.basename(path)}")
    
    return tif_paths


def handle_insufficient_images(images: List[np.ndarray], target_count: int) -> List[np.ndarray]:
    """Handle case when we have fewer images than required"""
    current_count = len(images)
    
    if current_count >= target_count:
        return images[:target_count]  # Truncate if we have more than needed
    
    if current_count == 0:
        raise ValueError("No images available to pad from")
    
    missing_count = target_count - current_count
    print(f"Warning: Only {current_count} images available, need {target_count} (missing {missing_count})")
    
    # Determine strategy - auto or specific
    strategy = CONFIG['padding_strategy']
    
    # Auto strategy selection based on number of missing images
    if strategy == 'auto':
        threshold = CONFIG['auto_padding_threshold']
        if missing_count <= threshold:
            strategy = 'interpolate'
            print(f"Auto strategy: Using 'interpolate' (missing images <= {threshold})")
        else:
            strategy = 'repeat_last'
            print(f"Auto strategy: Using 'repeat_last' (missing images > {threshold})")
    
    print(f"Applying padding strategy: {strategy}")
    
    if strategy == 'repeat_last':
        # Strategy 1: Repeat the last available image
        last_image = images[-1]
        while len(images) < target_count:
            images.append(last_image.copy())
        print(f"✓ Padded with {missing_count} copies of the last image")
    
    elif strategy == 'zero_pad':
        # Strategy 2: Zero padding
        if current_count > 0:
            zero_image = np.zeros_like(images[0])
            while len(images) < target_count:
                images.append(zero_image.copy())
            print(f"✓ Padded with {missing_count} zero images")
    
    elif strategy == 'interpolate':
        # Strategy 3: Linear interpolation between available images
        if current_count >= 2:
            # Interpolate between first and last image
            first_img = images[0]
            last_img = images[-1]
            
            # Calculate how many images we need to insert
            to_insert = target_count - current_count
            
            # Create and insert interpolated images
            for i in range(1, to_insert + 1):
                # Calculate alpha for this position
                alpha = i / (to_insert + 1)
                interpolated = (1 - alpha) * first_img + alpha * last_img
                images.append(interpolated)
            
            print(f"✓ Padded with {missing_count} interpolated images")
        else:
            # Fallback to repeat_last if only one image
            last_image = images[-1]
            while len(images) < target_count:
                images.append(last_image.copy())
            print(f"✓ Fallback: Padded with {missing_count} copies of the last image")
    
    else:
        raise ValueError(f"Unknown padding strategy: {strategy}")
    
    return images


def load_tif_images(tif_paths: List[str]) -> Tuple[List[np.ndarray], Dict, Tuple[int, int]]:
    """Load TIF images and extract geospatial info from first image"""
    print(f"Loading {len(tif_paths)} TIF images...")
    
    # Read geospatial info from first image
    geospatial_info = read_geospatial_info(tif_paths[0])
    
    images = []
    for filepath in tif_paths:
        with Image.open(filepath) as img:
            if img.mode != 'L':
                img = img.convert('L')
            data = np.array(img, dtype=np.float32)
            data = np.clip(data, 0, 100) * 0.01  # Normalize to [0, 1]
            images.append(data)
    
    # Handle insufficient images
    images = handle_insufficient_images(images, CONFIG['n_channels'])
    
    image_shape = images[0].shape
    print(f"✓ Loaded images with shape: {image_shape}")
    return images, geospatial_info, image_shape


def create_patches(images: List[np.ndarray], image_shape: Tuple[int, int]) -> Tuple[List[np.ndarray], List[Dict]]:
    """Create patches to cover full image with proper overlap handling"""
    H, W = image_shape
    patch_size = CONFIG['patch_size']
    
    print(f"Creating patches for image size: {H}x{W}, patch size: {patch_size}")
    
    # Calculate number of patches needed
    n_patches_h = max(1, (H + patch_size - 1) // patch_size)  # Ceiling division
    n_patches_w = max(1, (W + patch_size - 1) // patch_size)
    
    print(f"Will create {n_patches_h}x{n_patches_w} = {n_patches_h * n_patches_w} patches")
    
    positions = []
    patch_sequences = []
    
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Calculate start positions ensuring we don't go out of bounds
            start_h = min(i * patch_size, max(0, H - patch_size))
            start_w = min(j * patch_size, max(0, W - patch_size))
            
            # Ensure we don't exceed image boundaries
            end_h = min(start_h + patch_size, H)
            end_w = min(start_w + patch_size, W)
            
            # Skip if patch would be too small
            if (end_h - start_h) < patch_size // 2 or (end_w - start_w) < patch_size // 2:
                continue
            
            positions.append({
                'start_h': start_h,
                'start_w': start_w,
                'end_h': end_h,
                'end_w': end_w,
                'name': f'patch_{i}_{j}'
            })
            
            # Create patch sequence for all input images
            sequence = []
            for img in images:
                # Extract patch and pad if necessary
                patch = img[start_h:end_h, start_w:end_w]
                
                # Pad patch to full patch_size if needed
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    padded_patch = np.zeros((patch_size, patch_size), dtype=patch.dtype)
                    padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                    patch = padded_patch
                
                sequence.append(patch[:, :, np.newaxis])
            
            patch_sequences.append(np.array(sequence))
    
    print(f"✓ Created {len(patch_sequences)} patches")
    return patch_sequences, positions


def load_model(model_path: str) -> tf.keras.Model:
    """Load U-Net model"""
    print(f"Loading model from {model_path}...")
    
    try:
        from cloudcast.model import unet
    except ImportError:
        cloudcast_dir = os.path.join(workspace_path, 'cloudcast')
        if cloudcast_dir not in sys.path:
            sys.path.insert(0, cloudcast_dir)
        from model import unet
    
    # Create model with 9 input channels + 1 leadtime channel
    total_channels = CONFIG['n_channels'] + 1
    model = unet(
        pretrained_weights=None,
        input_size=(CONFIG['patch_size'], CONFIG['patch_size'], total_channels),
        loss_function="bcl1",
        optimizer="adam",
        compile=False
    )
    
    model.load_weights(model_path)
    print(f"✓ Model loaded successfully")
    return model


def predict_leadtime(model: tf.keras.Model, patches: List[np.ndarray], 
                    leadtime_idx: int) -> List[np.ndarray]:
    """Predict for a single leadtime"""
    patch_size = CONFIG['patch_size']
    batch_size = CONFIG['batch_size']
    leadtime_conditioning = CONFIG['leadtime_conditioning']
    
    # Create leadtime conditioning channel
    leadtime_channel = create_squeezed_leadtime_conditioning(
        (patch_size, patch_size), leadtime_conditioning, leadtime_idx
    )
    leadtime_channel = np.squeeze(leadtime_channel)
    leadtime_channel = np.expand_dims(leadtime_channel, axis=-1)
    
    # Prepare batch inputs
    batch_inputs = []
    for patch_seq in patches:
        # Stack all 9 input channels
        patch_stacked = np.concatenate([patch_seq[j] for j in range(len(patch_seq))], axis=-1)
        # Add leadtime channel
        patch_with_leadtime = np.concatenate([patch_stacked, leadtime_channel], axis=-1)
        batch_inputs.append(patch_with_leadtime)
    
    batch_array = np.array(batch_inputs)
    predictions = model.predict(batch_array, batch_size=batch_size, verbose=0)
    
    return [pred for pred in predictions]


def predict_parallel(model: tf.keras.Model, patches: List[np.ndarray], 
                    leadtimes: List[int]) -> Dict[int, List[np.ndarray]]:
    """Predict multiple leadtimes in parallel"""
    print(f"Predicting {len(leadtimes)} leadtimes in parallel...")
    
    results = {}
    
    # Use ThreadPoolExecutor for parallel leadtime prediction
    with ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        # Submit all leadtime predictions
        future_to_leadtime = {
            executor.submit(predict_leadtime, model, patches, lt): lt 
            for lt in leadtimes
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_leadtime):
            leadtime = future_to_leadtime[future]
            try:
                predictions = future.result()
                results[leadtime] = predictions
                print(f"✓ Completed leadtime {leadtime}")
            except Exception as e:
                print(f"✗ Error predicting leadtime {leadtime}: {e}")
                results[leadtime] = None
    
    return results


def assemble_patches(predictions: Dict[int, List[np.ndarray]], 
                    positions: List[Dict], leadtimes: List[int], 
                    image_shape: Tuple[int, int]) -> Dict[int, np.ndarray]:
    """Assemble patches back to full image with averaging in overlap region"""
    H, W = image_shape
    patch_size = CONFIG['patch_size']
    
    assembled_results = {}
    
    for leadtime in leadtimes:
        if predictions[leadtime] is None:
            continue
            
        # Initialize output and count arrays
        output = np.zeros((H, W), dtype=np.float64)
        counts = np.zeros((H, W), dtype=np.int32)
        
        # Add each patch to the output
        for pred, pos in zip(predictions[leadtime], positions):
            start_h, start_w = pos['start_h'], pos['start_w']
            end_h, end_w = pos['end_h'], pos['end_w']
            
            # Remove channel dimension if present
            if len(pred.shape) == 3:
                pred = pred[:, :, 0]
            
            # Get the actual size of the region to fill
            actual_h = end_h - start_h
            actual_w = end_w - start_w
            
            # Extract the relevant part of the prediction
            pred_region = pred[:actual_h, :actual_w]
            
            # Add to output and increment counts
            output[start_h:end_h, start_w:end_w] += pred_region
            counts[start_h:end_h, start_w:end_w] += 1
        
        # Average overlapping regions
        counts[counts == 0] = 1
        output = output / counts
        
        assembled_results[leadtime] = output
    
    print(f"✓ Assembled {len(assembled_results)} predictions")
    return assembled_results


def crop_predictions(predictions: Dict[int, np.ndarray], 
                    crop_margin: int, image_shape: Tuple[int, int]) -> Dict[int, np.ndarray]:
    """Crop predictions by removing margin pixels"""
    cropped_results = {}
    
    H, W = image_shape
    output_height = H - 2 * crop_margin
    output_width = W - 2 * crop_margin
    
    for leadtime, pred in predictions.items():
        # Ensure we don't crop beyond image boundaries
        if pred.shape[0] < 2 * crop_margin or pred.shape[1] < 2 * crop_margin:
            print(f"Warning: Image too small to crop {crop_margin}px margin. Using original size.")
            cropped_results[leadtime] = pred
        else:
            # Crop from all sides
            cropped = pred[crop_margin:crop_margin+output_height, crop_margin:crop_margin+output_width]
            cropped_results[leadtime] = cropped
    
    if cropped_results:
        actual_shape = cropped_results[list(cropped_results.keys())[0]].shape
        print(f"✓ Cropped predictions from {image_shape} to {actual_shape}")
    
    return cropped_results


def save_predictions(predictions: Dict[int, np.ndarray], output_dir: str, 
                    base_timestamp: str, geospatial_info: Dict, crop_margin: int):
    """Save predictions as TIF files with geospatial metadata and timestamp-based filenames"""
    from datetime import datetime, timedelta
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse base timestamp from the last input file
    # Expected format: YYYY-MM-DD-HH-MM
    try:
        base_dt = datetime.strptime(base_timestamp, '%Y-%m-%d-%H-%M')
    except ValueError:
        # Fallback: try to extract from filename if it contains timestamp
        import re
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})', base_timestamp)
        if timestamp_match:
            base_dt = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d-%H-%M')
        else:
            print(f"Warning: Could not parse timestamp from {base_timestamp}, using current time")
            base_dt = datetime.now()
    
    # Adjust geospatial info for cropped image
    if geospatial_info and 'geotransform' in geospatial_info:
        adjusted_geotransform = adjust_geotransform_for_crop(
            geospatial_info['geotransform'], crop_margin
        )
        geospatial_info = geospatial_info.copy()
        geospatial_info['geotransform'] = adjusted_geotransform
        
        # Update dimensions based on actual cropped size
        if predictions:
            sample_pred = list(predictions.values())[0]
            geospatial_info['height'], geospatial_info['width'] = sample_pred.shape
    
    saved_files = []
    
    for leadtime in sorted(predictions.keys()):
        pred = predictions[leadtime]
        
        # Calculate future timestamp based on leadtime (each leadtime = 10 minutes)
        future_dt = base_dt + timedelta(minutes=(leadtime + 1) * 10)
        timestamp_str = future_dt.strftime('%Y-%m-%d-%H-%M')
        
        # Scale to [0, 100] range for visualization
        pred_scaled = pred * 100.0
        
        # Create filename with timestamp format
        filename = f"{timestamp_str}.tif"
        filepath = os.path.join(output_dir, filename)
        
        # Save with geospatial info
        save_tif_with_geospatial_info(pred_scaled, filepath, geospatial_info)
        saved_files.append(filepath)
        
        print(f"✓ Saved {filename} | leadtime {leadtime} (+{(leadtime+1)*10}min) | range=[{pred.min():.4f}, {pred.max():.4f}] (model) | range=[{pred_scaled.min():.2f}, {pred_scaled.max():.2f}] (x100)")
    
    return saved_files


def main(end_timestamp: str = None, output_dir: str = 'production_output'):
    """Main production prediction pipeline
    
    Args:
        end_timestamp: Last timestamp in the sequence (e.g., '2025-10-27-08-40')
        output_dir: Directory to save predictions
    """
    
    # Default timestamp if not provided
    if end_timestamp is None:
        end_timestamp = '2025-10-27-08-40'  # Default example
    
    print("="*80)
    print("PRODUCTION RADAR PREDICTION")
    print("="*80)
    print(f"End timestamp:    {end_timestamp}")
    print(f"Required channels: {CONFIG['n_channels']}")
    print(f"Timestamp format: {CONFIG['timestamp_format']}")
    print(f"Data directory:   {CONFIG['data_directory']}")
    print(f"Time interval:    {CONFIG['time_interval_minutes']} minutes")
    print("="*80)
    
    # Auto-extract TIF paths based on end timestamp
    try:
        tif_paths = auto_extract_tif_paths(end_timestamp)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error extracting TIF paths: {e}")
        return
    
    base_timestamp = end_timestamp
    
    print(f"Input images:     {len(tif_paths)}")
    print(f"Base timestamp:   {base_timestamp}")
    print(f"Model path:       {CONFIG['model_path']}")
    print(f"Crop margin:      {CONFIG['crop_margin']}px")
    print(f"Leadtimes:        {CONFIG['leadtimes']}")
    print(f"Batch size:       {CONFIG['batch_size']}")
    print(f"Max workers:      {CONFIG['max_workers']}")
    print("="*80)
    
    try:
        # Load images and geospatial info
        images, geospatial_info, image_shape = load_tif_images(tif_paths)
        
        # Create patches
        patches, positions = create_patches(images, image_shape)
        
        # Load model
        model = load_model(CONFIG['model_path'])
        
        # Predict all leadtimes in parallel
        predictions = predict_parallel(model, patches, CONFIG['leadtimes'])
        
        # Assemble patches
        assembled = assemble_patches(predictions, positions, CONFIG['leadtimes'], image_shape)
        
        # Crop predictions
        cropped = crop_predictions(assembled, CONFIG['crop_margin'], image_shape)
        
        # Save results
        saved_files = save_predictions(cropped, output_dir, base_timestamp, geospatial_info, CONFIG['crop_margin'])
        
        print("\n" + "="*80)
        print("✅ PRODUCTION PREDICTION COMPLETED!")
        print(f"   Created {len(saved_files)} prediction files")
        print(f"   Output directory: {output_dir}")
        print(f"   Geospatial metadata: {'Preserved' if geospatial_info else 'Not available'}")
        print("="*80)
        
        # Memory cleanup
        del model, images, patches, predictions, assembled, cropped
        gc.collect()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Configure TensorFlow for CPU optimization
    tf.config.threading.set_intra_op_parallelism_threads(16)  # Use all CPU cores
    tf.config.threading.set_inter_op_parallelism_threads(8)   # Parallel operations
    
    # Command line argument support
    import sys
    if len(sys.argv) > 1:
        end_timestamp = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else 'production_output'
        main(end_timestamp, output_dir)
    else:
        main()
