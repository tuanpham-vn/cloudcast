import os
import numpy as np
import argparse
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from PIL import Image
import gc


def parse_command_line():
    parser = argparse.ArgumentParser(description="Radar nowcasting from TIF files")
    parser.add_argument("--timestamp", action="store", type=str, required=True,
                       help="Last timestamp in YYYY-MM-DD-HH-MM format (e.g., 2025-10-07-13-40)")
    parser.add_argument("--input_dir", action="store", type=str, default="data",
                       help="Directory containing input TIF files (default: data)")
    parser.add_argument("--output_dir", action="store", type=str, required=True,
                       help="Output directory for radar predictions")
    
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model_dir", action="store", type=str,
                       help="Path to model directory (checkpoint or saved model) or .h5 weights file")
    model_group.add_argument("--weights_file", action="store", type=str,
                       help="Path to .h5 weights file (deprecated, use --model_dir instead)")
    
    parser.add_argument("--patch_size", action="store", type=int, default=512,
                       help="Patch size for sliding window (default: 512)")
    parser.add_argument("--overlap_ratio", action="store", type=float, default=0.1,
                       help="Overlap ratio between patches (default: 0.1)")
    parser.add_argument("--center_crop_ratio", action="store", type=float, default=0.6,
                       help="Center crop ratio for output patches (default: 0.6, range: 0.0-1.0)")
    parser.add_argument("--n_forecast", action="store", type=int, default=18,
                       help="Number of forecast frames (default: 18)")
    parser.add_argument("--n_channels", action="store", type=int, default=None,
                       help="Number of input data channels (without leadtime conditioning)")
    parser.add_argument("--leadtime_conditioning", action="store", type=int, default=18,
                       help="Leadtime conditioning depth (default: 18)")
    parser.add_argument("--batch_size", action="store", type=int, default=32,
                       help="Batch size for prediction (default: 32)")
    parser.add_argument("--create_colored_tif", action="store_true",
                       help="Create colored TIF files (default: False)")
    parser.add_argument("--create_png", action="store_true",
                       help="Create PNG files from predictions (default: False)")
    parser.add_argument("--show_progress", action="store_true",
                       help="Show progress during processing")
    parser.add_argument("--sequence_stride_minutes", action="store", type=int, default=20,
                       help="Stride in minutes between predictions (default: 20, use 10 for 10-minute models)")
    parser.add_argument("--force_n_channels", action="store", type=int, default=None,
                       help="Force specific TOTAL number of channels including leadtime (to fix tensor shape mismatch)")
    parser.add_argument("--clip_values", action="store_true", default=True,
                       help="Clip input and output values to [0, 100] range (default: True)")
    
    args = parser.parse_args()
    return args


def get_sequential_tif_files(timestamp: str, input_dir: str, n_input: int = 6, 
                             time_interval: int = 10, sequence_stride_minutes: int = 10,
                             force_n_input: int = None) -> List[str]:
    """
    Get sequential TIF files with specified time interval
    
    Args:
        timestamp: Format 'YYYY-MM-DD-HH-MM'
        input_dir: Directory containing TIF files
        n_input: Number of input files (default: 6)
        time_interval: Time interval in minutes (default: 10)
        sequence_stride_minutes: Stride in minutes between files (default: 10, use 20 for 20-min models)
        force_n_input: Force specific number of input files (overrides n_input)
    
    Returns:
        List of file paths from oldest to newest
    """
    dt = datetime.strptime(timestamp, '%Y-%m-%d-%H-%M')
    files = []
    
    # Use forced number of inputs if specified
    actual_n_input = force_n_input if force_n_input is not None else n_input
    
    # Calculate total time span needed
    total_minutes = (actual_n_input - 1) * sequence_stride_minutes
    
    # Start from the earliest timestamp needed
    for i in range(actual_n_input-1, -1, -1):
        dt_i = dt - timedelta(minutes=i * sequence_stride_minutes)
        timestamp_i = dt_i.strftime('%Y-%m-%d_%H%M')
        filename = f"{timestamp_i}_acc10.tif"
        filepath = os.path.join(input_dir, filename)
        files.append(filepath)
    
    return files


def read_tif_file(filepath: str, clip_values: bool = True) -> np.ndarray:
    """
    Read TIF file and return normalized array
    
    Args:
        filepath: Path to TIF file
        clip_values: Whether to clip values to [0, 100] range
        
    Returns:
        Normalized array with values in [0, 1] range
    """
    try:
        with Image.open(filepath) as img:
            if img.mode != 'L':
                img = img.convert('L')
            data = np.array(img, dtype=np.float32) / 255.0
            
            # Scale to [0, 100] range and clip if requested
            data = data * 100.0
            if clip_values:
                data = np.clip(data, 0, 100)
                
            # Normalize back to [0, 1] for model input
            data = data / 100.0
            
            return data
    except Exception as e:
        raise RuntimeError(f"Error reading {filepath}: {e}")


def extract_patches_sliding_window(image: np.ndarray, patch_size: Tuple[int, int], 
                                   overlap_ratio: float) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Extract patches using sliding window with overlap (same logic as create_tiff_dataset.py)
    
    Args:
        image: Input image (H, W) or (H, W, C)
        patch_size: Size of each patch (H, W)
        overlap_ratio: Overlap ratio (0.0 to 1.0)
    
    Returns:
        patches: List of patch arrays
        positions: List of position metadata
    """
    if len(image.shape) == 2:
        H, W = image.shape
        image = image[:, :, np.newaxis]
    else:
        H, W, C = image.shape
    
    patch_h, patch_w = patch_size
    stride_h = int(patch_h * (1 - overlap_ratio))
    stride_w = int(patch_w * (1 - overlap_ratio))
    
    patches = []
    positions = []
    
    n_patches_h = max(1, (H - patch_h) // stride_h + 1)
    n_patches_w = max(1, (W - patch_w) // stride_w + 1)
    
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            start_h = min(i * stride_h, H - patch_h)
            start_w = min(j * stride_w, W - patch_w)
            
            patch = image[start_h:start_h + patch_h, start_w:start_w + patch_w, :]
            
            if patch.shape[0] == patch_h and patch.shape[1] == patch_w:
                patches.append(patch)
                positions.append({
                    'start_h': start_h,
                    'start_w': start_w,
                    'center_h': start_h + patch_h // 2,
                    'center_w': start_w + patch_w // 2
                })
    
    return patches, positions


def load_and_extract_patches(tif_files: List[str], patch_size: int, 
                             overlap_ratio: float, clip_values: bool = True) -> Tuple[List[np.ndarray], List[Dict], Tuple[int, int]]:
    """
    Load TIF files and extract patches
    
    Args:
        tif_files: List of TIF file paths
        patch_size: Size of each patch
        overlap_ratio: Overlap ratio between patches
        clip_values: Whether to clip values to [0, 100] range
    
    Returns:
        patch_sequences: List of (n_input, H, W, 1) arrays
        positions: List of position metadata
        image_shape: Original image shape (H, W)
    """
    print(f"\nLoading {len(tif_files)} TIF files...")
    
    images = []
    for filepath in tif_files:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        data = read_tif_file(filepath, clip_values=clip_values)
        images.append(data)
        print(f"  ✓ {os.path.basename(filepath)}")
    
    image_shape = images[0].shape
    print(f"  Image shape: {image_shape}")
    
    print(f"\nExtracting patches (size={patch_size}, overlap={overlap_ratio})...")
    
    first_patches, positions = extract_patches_sliding_window(
        images[0], (patch_size, patch_size), overlap_ratio
    )
    n_patches = len(first_patches)
    
    patch_sequences = []
    for patch_idx in range(n_patches):
        pos = positions[patch_idx]
        start_h, start_w = pos['start_h'], pos['start_w']
        
        sequence = []
        for img in images:
            patch = img[start_h:start_h+patch_size, start_w:start_w+patch_size]
            sequence.append(patch[:, :, np.newaxis])
        
        patch_sequences.append(np.array(sequence))
    
    print(f"  ✓ Extracted {n_patches} patches")
    
    return patch_sequences, positions, image_shape


def load_model_auto(model_path: str, patch_h: int, patch_w: int, n_channels: int, force_n_channels: int = None):
    """
    Auto-detect and load model from file or directory
    
    Args:
        model_path: Path to model (can be .h5 file, checkpoint dir, or saved model dir)
        patch_h, patch_w: Patch dimensions
        n_channels: Number of input channels (NOT including leadtime conditioning)
        force_n_channels: Force specific number of channels when loading model
    
    Returns:
        Loaded model
    """
    import tensorflow as tf
    import sys
    import os
    
    # Add project root to path to ensure imports work
    workspace_path = os.path.dirname(os.path.abspath(__file__))
    if workspace_path not in sys.path:
        sys.path.insert(0, workspace_path)
    
    # Try different import paths for model
    try:
        from cloudcast.model import unet
    except ImportError:
        try:
            # Add cloudcast directory to path
            cloudcast_dir = os.path.join(workspace_path, 'cloudcast')
            if cloudcast_dir not in sys.path:
                sys.path.insert(0, cloudcast_dir)
            from model import unet
        except ImportError:
            raise ImportError("Could not import unet model from cloudcast.model or model")
    
    # Use forced channels if specified (to fix tensor shape mismatch)
    # If force_n_channels is provided, it represents the TOTAL number of channels including leadtime
    if force_n_channels is not None:
        # force_n_channels already includes leadtime conditioning
        total_channels = force_n_channels
        print(f"  → Creating model with input shape: ({patch_h}, {patch_w}, {total_channels})")
        print(f"  → Using {total_channels} total channels (forced)")
    else:
        # Add 1 for leadtime conditioning
        total_channels = n_channels + 1
        print(f"  → Creating model with input shape: ({patch_h}, {patch_w}, {total_channels})")
        print(f"  → Using {n_channels} data channels + 1 leadtime channel = {total_channels} total")
    
    # Case 1: H5 weights file
    if model_path.endswith('.h5') or model_path.endswith('.weights.h5'):
        print(f"  → Loading H5 weights file...")
        model = unet(
            pretrained_weights=None,
            input_size=(patch_h, patch_w, total_channels),
            loss_function="bcl1",
            optimizer="adam",
            compile=False
        )
        
        try:
            model.load_weights(model_path)
            print(f"  ✓ Model weights loaded successfully")
            return model
        except ValueError as e:
            print(f"\n❌ Error loading weights: {e}")
            print("\nThis error often occurs when the model was trained with a different number of channels.")
            print("Try using --force_n_channels to match the model's expected input shape.")
            print("Example: If model expects 5 total channels (4 data + 1 leadtime), use --force_n_channels 5")
            raise
    
    # Case 2: Checkpoint file (.ckpt)
    if model_path.endswith('.ckpt'):
        if os.path.exists(model_path + ".index"):
            print(f"  → Loading checkpoint file...")
            model = unet(
                pretrained_weights=None,
                input_size=(patch_h, patch_w, total_channels),
                loss_function="bcl1",
                optimizer="adam",
                compile=False
            )
            
            try:
                model.load_weights(model_path)
                print(f"  ✓ Checkpoint loaded successfully")
                return model
            except ValueError as e:
                print(f"\n❌ Error loading weights: {e}")
                print("\nThis error often occurs when the model was trained with a different number of channels.")
                print("Try using --force_n_channels to match the model's expected input shape.")
                print("Example: If model expects 5 total channels (4 data + 1 leadtime), use --force_n_channels 5")
                raise
    
    # Case 3: Directory (checkpoint or saved model)
    if os.path.isdir(model_path):
        # Try checkpoint first
        checkpoint_path = os.path.join(model_path, "cp.ckpt")
        weights_path = os.path.join(model_path, "cp.weights.h5")
        
        if os.path.exists(checkpoint_path + ".index") or os.path.exists(weights_path):
            print(f"  → Loading from checkpoint directory...")
            # Use cp.weights.h5 if exists, otherwise cp.ckpt
            if os.path.exists(weights_path):
                checkpoint_file = weights_path
            else:
                checkpoint_file = checkpoint_path
            
            model = unet(
                pretrained_weights=None,
                input_size=(patch_h, patch_w, total_channels),
                loss_function="bcl1",
                optimizer="adam",
                compile=False
            )
            
            try:
                model.load_weights(checkpoint_file)
                print(f"  ✓ Checkpoint loaded successfully")
                return model
            except ValueError as e:
                print(f"\n❌ Error loading weights: {e}")
                print("\nThis error often occurs when the model was trained with a different number of channels.")
                print("Try using --force_n_channels to match the model's expected input shape.")
                print("Example: If model expects 5 total channels (4 data + 1 leadtime), use --force_n_channels 5")
                raise
        
        # Try saved model
        try:
            print(f"  → Loading saved model directory...")
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"  ✓ Saved model loaded successfully")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load from directory {model_path}: {e}")
    
    raise RuntimeError(f"Unknown model path format: {model_path}")


def predict_with_model(model_path: str, patches: List[np.ndarray], 
                      batch_size: int, n_forecast: int, n_channels: int,
                      leadtime_conditioning: int, show_progress: bool = False,
                      sequence_stride_minutes: int = 10, force_n_channels: int = None,
                      clip_values: bool = True) -> List[np.ndarray]:
    """
    Load model and predict on patches in batches
    
    Args:
        model_path: Path to model (H5 file, checkpoint dir, or saved model dir)
        patches: List of input patch sequences (n_input, H, W, 1)
        batch_size: Batch size for prediction
        n_forecast: Number of forecast frames
        n_channels: Number of input channels (without leadtime)
        leadtime_conditioning: Leadtime conditioning depth
        show_progress: Show progress bar
        sequence_stride_minutes: Stride in minutes between predictions (10 or 20)
        force_n_channels: Force specific number of channels when loading model
        clip_values: Whether to clip values to [0, 100] range
    
    Returns:
        List of predictions (n_forecast, H, W, 1)
    """
    print(f"\nLoading model from {model_path}...")
    
    try:
        import sys
        import os
        workspace_path = os.path.dirname(os.path.abspath(__file__))
        if workspace_path not in sys.path:
            sys.path.insert(0, workspace_path)
        
        import tensorflow as tf
        from base.preprocess import create_squeezed_leadtime_conditioning
        
        patch_h, patch_w = patches[0].shape[1:3]
        
        # Load model with appropriate number of channels
        model = load_model_auto(
            model_path, 
            patch_h, 
            patch_w, 
            n_channels,  # n_channels is without leadtime conditioning
            force_n_channels=force_n_channels  # force_n_channels includes leadtime if specified
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    # Calculate stride factor (1 for 10-min, 2 for 20-min)
    stride_factor = sequence_stride_minutes // 10
    if stride_factor < 1:
        stride_factor = 1
        print(f"Warning: Invalid stride minutes {sequence_stride_minutes}, using 10 minutes")
    
    print(f"\nPredicting {len(patches)} patches x {n_forecast} leadtimes (batch_size={batch_size})...")
    print(f"  Model input shape: {model.input_shape}")
    print(f"  Model output shape: {model.output_shape}")
    print(f"  Using {sequence_stride_minutes}-minute stride (factor: {stride_factor})")
    
    all_predictions = []
    
    for leadtime_idx in range(n_forecast):
        # Calculate actual leadtime index based on stride
        actual_lt = leadtime_idx * stride_factor
        
        print(f"\n  Leadtime {leadtime_idx+1}/{n_forecast} (+{(leadtime_idx+1)*sequence_stride_minutes} min)...")
        print(f"    Using actual leadtime index: {actual_lt}")
        
        leadtime_channel = create_squeezed_leadtime_conditioning(
            (patch_h, patch_w), leadtime_conditioning, actual_lt
        )
        leadtime_channel = np.squeeze(leadtime_channel)
        leadtime_channel = np.expand_dims(leadtime_channel, axis=-1)
        
        leadtime_predictions = []
        n_batches = (len(patches) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(patches))
            
            batch_inputs = []
            for i in range(start_idx, end_idx):
                patch_seq = patches[i]
                
                # Use all channels or forced number of channels
                if force_n_channels is not None and force_n_channels < len(patch_seq):
                    # Use only the first force_n_channels
                    patch_stacked = np.concatenate([patch_seq[j] for j in range(force_n_channels)], axis=-1)
                else:
                    # Use all available channels
                    patch_stacked = np.concatenate([patch_seq[j] for j in range(len(patch_seq))], axis=-1)
                
                patch_with_leadtime = np.concatenate([patch_stacked, leadtime_channel], axis=-1)
                batch_inputs.append(patch_with_leadtime)
            
            batch_array = np.array(batch_inputs)
            batch_preds = model.predict(batch_array, verbose=0)
            
            # Clip predictions to [0, 100] range if requested
            if clip_values:
                batch_preds = np.clip(batch_preds * 100.0, 0, 100) / 100.0
            
            for pred in batch_preds:
                leadtime_predictions.append(pred)
            
            if show_progress:
                processed = end_idx
                total = len(patches)
                print(f"    Batch {batch_idx+1}/{n_batches}: {processed}/{total} patches", end="\r")
        
        if show_progress:
            print()
        
        all_predictions.append(leadtime_predictions)
        
        if (leadtime_idx + 1) % 5 == 0:
            gc.collect()
    
    combined_predictions = []
    for patch_idx in range(len(patches)):
        patch_preds = []
        for leadtime_idx in range(n_forecast):
            pred = all_predictions[leadtime_idx][patch_idx]
            patch_preds.append(pred)
        
        patch_preds_array = np.array(patch_preds)
        combined_predictions.append(patch_preds_array)
    
    print(f"\n  ✓ Prediction completed")
    
    return combined_predictions


def assemble_predictions(predictions: List[np.ndarray], positions: List[Dict],
                        image_shape: Tuple[int, int], n_forecast: int,
                        center_crop_ratio: float) -> List[np.ndarray]:
    """
    Assemble predictions by placing center crops at patch centers
    
    Args:
        predictions: List of (n_forecast, H, W, 1) predictions
        positions: List of position metadata
        image_shape: Original image shape (H, W)
        n_forecast: Number of forecast frames
        center_crop_ratio: Ratio of center crop to use (0.0-1.0)
    
    Returns:
        List of assembled frames (H, W)
    """
    print(f"\nAssembling predictions (center_crop_ratio={center_crop_ratio})...")
    
    H, W = image_shape
    patch_size = predictions[0].shape[1]
    crop_size = int(patch_size * center_crop_ratio)
    crop_offset = (patch_size - crop_size) // 2
    
    outputs = [np.zeros((H, W), dtype=np.float64) for _ in range(n_forecast)]
    counts = np.zeros((H, W), dtype=np.int32)
    
    for pred, pos in zip(predictions, positions):
        center_h, center_w = pos['center_h'], pos['center_w']
        
        crop_start_h = max(0, center_h - crop_size // 2)
        crop_start_w = max(0, center_w - crop_size // 2)
        crop_end_h = min(H, crop_start_h + crop_size)
        crop_end_w = min(W, crop_start_w + crop_size)
        
        actual_h = crop_end_h - crop_start_h
        actual_w = crop_end_w - crop_start_w
        
        for t in range(n_forecast):
            if len(pred.shape) == 4:
                frame = pred[t, crop_offset:crop_offset+actual_h, crop_offset:crop_offset+actual_w, 0]
            else:
                frame = pred[t, crop_offset:crop_offset+actual_h, crop_offset:crop_offset+actual_w]
            outputs[t][crop_start_h:crop_end_h, crop_start_w:crop_end_w] += frame
        
        counts[crop_start_h:crop_end_h, crop_start_w:crop_end_w] += 1
    
    counts[counts == 0] = 1
    
    for t in range(n_forecast):
        outputs[t] = outputs[t] / counts
    
    print(f"  ✓ Assembled {n_forecast} frames")
    
    return outputs


def save_predictions(outputs: List[np.ndarray], output_dir: str, 
                    base_timestamp: str, sequence_stride_minutes: int = 10,
                    clip_values: bool = True):
    """
    Save predictions as TIF files
    
    Args:
        outputs: List of prediction frames
        output_dir: Output directory
        base_timestamp: Base timestamp string
        sequence_stride_minutes: Stride in minutes between predictions
        clip_values: Whether to clip values to [0, 100] range
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving {len(outputs)} TIF files to {output_dir}...")
    print(f"  Using {sequence_stride_minutes}-minute time stride")
    
    dt_base = datetime.strptime(base_timestamp, '%Y-%m-%d-%H-%M')
    output_files = []
    
    for t, frame in enumerate(outputs):
        # Calculate timestamp based on stride minutes
        dt_out = dt_base + timedelta(minutes=(t+1) * sequence_stride_minutes)
        timestamp_out = dt_out.strftime('%Y-%m-%d_%H%M')
        filename = f"radar_{timestamp_out}_pred.tif"
        filepath = os.path.join(output_dir, filename)
        
        # Clip values if requested (already normalized to 0-1 range)
        if clip_values:
            frame = np.clip(frame, 0, 1)
            
        # Convert to 8-bit image and save
        img = Image.fromarray((frame * 255).astype(np.uint8), mode='L')
        img.save(filepath)
        
        output_files.append(filepath)
        print(f"  ✓ {filename} | range=[{frame.min():.4f}, {frame.max():.4f}]")
    
    return output_files


def create_colored_tif(grayscale_tif_path: str, output_dir: str, 
                      timestamp_str: str) -> str:
    """
    Create colored TIF from grayscale TIF
    
    Args:
        grayscale_tif_path: Path to grayscale TIF
        output_dir: Output directory for colored TIF
        timestamp_str: Timestamp string for filename
    
    Returns:
        Path to colored TIF file
    """
    from color_utils import create_cloud_colormap, apply_cloud_colors
    
    colored_dir = os.path.join(output_dir, "colored_tif")
    os.makedirs(colored_dir, exist_ok=True)
    
    img = Image.open(grayscale_tif_path)
    data = np.array(img, dtype=np.float32) / 255.0
    
    cmap, norm, boundaries = create_cloud_colormap()
    colored_data = apply_cloud_colors(data, cmap, norm, normalize=True)
    
    colored_data_uint8 = (colored_data * 255).astype(np.uint8)
    colored_img = Image.fromarray(colored_data_uint8, mode='RGBA')
    
    output_filename = f"radar_{timestamp_str}_pred_colored.tif"
    output_path = os.path.join(colored_dir, output_filename)
    colored_img.save(output_path)
    
    return output_path


def create_png_from_tif(tif_path: str, output_dir: str, 
                       timestamp_str: str, is_colored: bool = False) -> str:
    """
    Create PNG from TIF file
    
    Args:
        tif_path: Path to TIF file (grayscale or colored)
        output_dir: Output directory for PNG
        timestamp_str: Timestamp string for filename
        is_colored: Whether input is colored TIF
    
    Returns:
        Path to PNG file
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from color_utils import create_cloud_colormap, apply_cloud_colors
    
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
    
    img = Image.open(tif_path)
    data = np.array(img)
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    if is_colored:
        ax.imshow(data)
    else:
        data_normalized = data.astype(np.float32) / 255.0
        cmap, norm, boundaries = create_cloud_colormap()
        colored_data = apply_cloud_colors(data_normalized, cmap, norm, normalize=True)
        ax.imshow(colored_data)
    
    ax.axis('off')
    ax.set_title(f"Radar Prediction: {timestamp_str}", fontsize=14, fontweight='bold')
    
    output_filename = f"radar_{timestamp_str}_pred.png"
    output_path = os.path.join(png_dir, output_filename)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    return output_path


def create_visualization_outputs(tif_files: List[str], output_dir: str,
                                create_colored: bool = False, 
                                create_png: bool = False) -> Dict[str, List[str]]:
    """
    Create colored TIF and PNG outputs from grayscale TIF predictions
    
    Args:
        tif_files: List of grayscale TIF file paths
        output_dir: Base output directory
        create_colored: Whether to create colored TIF files
        create_png: Whether to create PNG files
    
    Returns:
        Dictionary with lists of created file paths
    """
    results = {
        'colored_tif': [],
        'png': []
    }
    
    if not create_colored and not create_png:
        return results
    
    print(f"\nCreating visualization outputs...")
    
    for tif_path in tif_files:
        basename = os.path.basename(tif_path)
        timestamp_str = basename.replace('radar_', '').replace('_pred.tif', '')
        
        try:
            if create_colored:
                colored_path = create_colored_tif(tif_path, output_dir, timestamp_str)
                results['colored_tif'].append(colored_path)
                print(f"  ✓ Colored TIF: {os.path.basename(colored_path)}")
            
            if create_png:
                if create_colored:
                    png_path = create_png_from_tif(colored_path, output_dir, 
                                                   timestamp_str, is_colored=True)
                else:
                    png_path = create_png_from_tif(tif_path, output_dir, 
                                                   timestamp_str, is_colored=False)
                results['png'].append(png_path)
                print(f"  ✓ PNG: {os.path.basename(png_path)}")
        
        except Exception as e:
            print(f"  ✗ Error processing {basename}: {e}")
    
    return results


def main():
    args = parse_command_line()
    
    # Determine model path
    model_path = args.model_dir if args.model_dir else args.weights_file
    
    # Calculate number of input channels
    # If force_n_channels is specified, it includes leadtime, so data channels = force_n_channels - 1
    # If n_channels is specified, use it directly
    # Otherwise, default to 6 input files
    if args.force_n_channels is not None:
        n_data_channels = args.force_n_channels - 1  # Subtract 1 for leadtime conditioning
        force_n_input = n_data_channels
    elif args.n_channels is not None:
        n_data_channels = args.n_channels
        force_n_input = n_data_channels
    else:
        n_data_channels = 6  # Default
        force_n_input = None
    
    print("="*80)
    print("RADAR NOWCASTING FROM TIF FILES")
    print("="*80)
    print(f"Timestamp:             {args.timestamp}")
    print(f"Input directory:       {args.input_dir}")
    print(f"Output directory:      {args.output_dir}")
    print(f"Model path:            {model_path}")
    print(f"Patch size:            {args.patch_size}x{args.patch_size}")
    print(f"Overlap ratio:         {args.overlap_ratio}")
    print(f"Center crop ratio:     {args.center_crop_ratio}")
    print(f"Forecast frames:       {args.n_forecast}")
    print(f"Input data channels:   {n_data_channels}")
    print(f"Leadtime conditioning: {args.leadtime_conditioning}")
    print(f"Batch size:            {args.batch_size}")
    print(f"Sequence stride:       {args.sequence_stride_minutes} minutes")
    if args.force_n_channels:
        print(f"Force total channels:  {args.force_n_channels} (data={n_data_channels} + leadtime=1)")
    print(f"Clip values:           {'Yes' if args.clip_values else 'No'}")
    print("="*80)
    
    try:
        tif_files = get_sequential_tif_files(
            args.timestamp, 
            args.input_dir,
            sequence_stride_minutes=args.sequence_stride_minutes,
            force_n_input=force_n_input
        )
        n_input_frames = len(tif_files)
        
        patch_sequences, positions, image_shape = load_and_extract_patches(
            tif_files, 
            args.patch_size, 
            args.overlap_ratio,
            clip_values=args.clip_values
        )
        
        predictions = predict_with_model(
            model_path, 
            patch_sequences, 
            args.batch_size, 
            args.n_forecast, 
            n_input_frames, 
            args.leadtime_conditioning,
            args.show_progress,
            sequence_stride_minutes=args.sequence_stride_minutes,
            force_n_channels=args.force_n_channels,
            clip_values=args.clip_values
        )
        
        assembled = assemble_predictions(
            predictions, 
            positions, 
            image_shape, 
            args.n_forecast, 
            args.center_crop_ratio
        )
        
        output_files = save_predictions(
            assembled, 
            args.output_dir, 
            args.timestamp,
            sequence_stride_minutes=args.sequence_stride_minutes,
            clip_values=args.clip_values
        )
        
        viz_results = create_visualization_outputs(
            output_files, 
            args.output_dir,
            create_colored=args.create_colored_tif,
            create_png=args.create_png
        )
        
        print("\n" + "="*80)
        print(f"✅ COMPLETED SUCCESSFULLY!")
        print(f"   Created {len(output_files)} radar prediction TIF files")
        print(f"   Time stride: {args.sequence_stride_minutes} minutes")
        if viz_results['colored_tif']:
            print(f"   Created {len(viz_results['colored_tif'])} colored TIF files")
            print(f"   Colored TIF directory: {os.path.join(args.output_dir, 'colored_tif')}")
        if viz_results['png']:
            print(f"   Created {len(viz_results['png'])} PNG files")
            print(f"   PNG directory: {os.path.join(args.output_dir, 'png')}")
        print(f"   Output directory: {args.output_dir}")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nRequired files:")
        files = get_sequential_tif_files(
            args.timestamp, 
            args.input_dir,
            sequence_stride_minutes=args.sequence_stride_minutes
        )
        for f in files:
            status = "✓" if os.path.exists(f) else "✗"
            print(f"  {status} {f}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# ========================================
# EXAMPLE COMMANDS
# ========================================
#
# 1. Basic prediction với 20-minute model (default):
# python predict_radar_from_tif.py \
#   --timestamp "2025-10-07-13-40" \
#   --input_dir "data" \
#   --output_dir "predictions" \
#   --model_dir "models/my_20min_model"
#
# 2. Prediction với custom parameters:
# python predict_radar_from_tif.py \
#   --timestamp "2025-10-07-13-40" \
#   --input_dir "data" \
#   --output_dir "predictions" \
#   --model_dir "models/my_model" \
#   --patch_size 256 \
#   --overlap_ratio 0.2 \
#   --n_forecast 12 \
#   --sequence_stride_minutes 20
#
# 3. Prediction với 10-minute model:
# python predict_radar_from_tif.py \
#   --timestamp "2025-10-07-13-40" \
#   --input_dir "data" \
#   --output_dir "predictions_10min" \
#   --model_dir "models/my_10min_model" \
#   --sequence_stride_minutes 10 \
#   --n_channels 6
#
# 4. Prediction với visualization outputs:
# python predict_radar_from_tif.py \
#   --timestamp "2025-10-07-13-40" \
#   --input_dir "data" \
#   --output_dir "predictions" \
#   --model_dir "models/my_model" \
#   --create_colored_tif \
#   --create_png \
#   --show_progress
#
# 5. Prediction với specific number of channels (fix tensor mismatch):
# python predict_radar_from_tif.py \
#   --timestamp "2025-10-07-13-40" \
#   --input_dir "data" \
#   --output_dir "predictions" \
#   --model_dir "models/my_model" \
#   --force_n_channels 5 \
#   --sequence_stride_minutes 20
#
# 6. Prediction với checkpoint weights file:
# python predict_radar_from_tif.py \
#   --timestamp "2025-10-07-13-40" \
#   --input_dir "data" \
#   --output_dir "predictions" \
#   --model_dir "checkpoints/my_model/cp.ckpt" \
#   --n_channels 4 \
#   --sequence_stride_minutes 20
#

