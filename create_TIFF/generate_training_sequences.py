#!/usr/bin/env python3
"""
Script to generate specific file lists for training from timestamp sequences
"""

import os
import glob
import re
from datetime import datetime, timedelta
import json
import sys

def extract_timestamp_from_filename(filename):
    """Extract timestamp from filename YYYY-MM-DD_HHMM_acc10.tif"""
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{4})_acc\d+\.tif', filename)
    if match:
        return match.group(1)
    return None

def parse_timestamp(ts_str):
    """Parse timestamp string to datetime object"""
    return datetime.strptime(ts_str, '%Y-%m-%d_%H%M')

def generate_training_sequences(timestamps, history_frames=4, prediction_frames=1, max_gap=2):
    """
    Generate training sequences from timestamps
    history_frames: number of history frames required
    prediction_frames: number of prediction frames
    max_gap: maximum allowed gap between frames
    """
    sequences = []
    current_sequence = []
    current_gaps = 0
    
    for i, ts in enumerate(timestamps):
        if not current_sequence:
            # Start new sequence
            current_sequence = [ts]
            current_gaps = 0
        else:
            # Check gap with previous frame
            prev_time = parse_timestamp(current_sequence[-1])
            curr_time = parse_timestamp(ts)
            expected_time = prev_time + timedelta(minutes=10)
            
            if curr_time == expected_time:
                # Perfect continuity
                current_sequence.append(ts)
            elif curr_time > expected_time:
                # Has gap
                gap_minutes = (curr_time - expected_time).total_seconds() / 60
                gap_count = int(gap_minutes / 10)
                
                if gap_count <= max_gap:
                    current_gaps += gap_count
                    current_sequence.append(ts)
                else:
                    # Gap too large, end current sequence
                    if len(current_sequence) >= history_frames + prediction_frames:
                        sequences.append({
                            'timestamps': current_sequence.copy(),
                            'length': len(current_sequence),
                            'gaps': current_gaps,
                            'completeness': ((len(current_sequence) - current_gaps) / len(current_sequence)) * 100
                        })
                    current_sequence = [ts]
                    current_gaps = 0
    
    # Add final sequence
    if len(current_sequence) >= history_frames + prediction_frames:
        sequences.append({
            'timestamps': current_sequence,
            'length': len(current_sequence),
            'gaps': current_gaps,
            'completeness': ((len(current_sequence) - current_gaps) / len(current_sequence)) * 100
        })
    
    return sequences

def create_training_samples(sequence, history_frames=4, prediction_frames=1):
    """Create training samples from a sequence"""
    samples = []
    timestamps = sequence['timestamps']
    
    for i in range(len(timestamps) - history_frames - prediction_frames + 1):
        # Get history frames
        history = timestamps[i:i + history_frames]
        
        # Get prediction frame
        prediction = timestamps[i + history_frames + prediction_frames - 1]
        
        # Create sample
        sample = {
            'history_timestamps': history,
            'prediction_timestamp': prediction,
            'history_files': [f"{ts}_acc10.tif" for ts in history],
            'prediction_file': f"{prediction}_acc10.tif",
            'sample_index': i
        }
        samples.append(sample)
    
    return samples

def analyze_tiff_training_data(directory="tiff_files", output_file="training_sequences.json"):
    """Analyze and generate training data from TIF files"""
    
    # Find all TIF files
    pattern = os.path.join(directory, "*.tif")
    tif_files = glob.glob(pattern)
    
    if not tif_files:
        print(f"❌ No TIF files found in {directory}")
        return
    
    print(f"📁 Analyzing training data from: {directory}")
    print(f"📊 Total TIF files: {len(tif_files)}")
    print("="*80)
    
    # Extract timestamps
    timestamps = []
    for file_path in tif_files:
        filename = os.path.basename(file_path)
        ts = extract_timestamp_from_filename(filename)
        if ts:
            timestamps.append(ts)
    
    timestamps.sort()
    print(f"✅ Found {len(timestamps)} files with valid timestamps")
    
    # Generate training sequences
    print("\n🔗 Generating training sequences...")
    sequences = generate_training_sequences(timestamps, history_frames=4, prediction_frames=1, max_gap=2)
    
    if not sequences:
        print("❌ No sequences found suitable for training")
        return
    
    print(f"✅ Found {len(sequences)} suitable sequences")
    
    # Create training samples
    all_samples = []
    for i, sequence in enumerate(sequences, 1):
        print(f"\nSequence {i}: {sequence['length']} frames, {sequence['gaps']} gaps, {sequence['completeness']:.1f}% complete")
        
        samples = create_training_samples(sequence, history_frames=4, prediction_frames=1)
        print(f"  → Generated {len(samples)} training samples")
        
        all_samples.extend(samples)
    
    print(f"\n📊 Total training samples: {len(all_samples)}")
    
    # Save results
    result = {
        'metadata': {
            'total_tif_files': len(tif_files),
            'total_timestamps': len(timestamps),
            'total_sequences': len(sequences),
            'total_training_samples': len(all_samples),
            'history_frames': 4,
            'prediction_frames': 1,
            'max_gap': 2,
            'analysis_time': datetime.now().isoformat()
        },
        'sequences': sequences,
        'training_samples': all_samples
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_file}")
    
    # Display sample examples
    print(f"\n📋 SAMPLE TRAINING SAMPLES (first 5):")
    print("-" * 60)
    for i, sample in enumerate(all_samples[:5], 1):
        print(f"Sample {i}:")
        print(f"  History: {', '.join(sample['history_timestamps'])}")
        print(f"  Prediction: {sample['prediction_timestamp']}")
        print(f"  Files: {', '.join(sample['history_files'])} → {sample['prediction_file']}")
        print()
    
    # Time-based statistics
    print("📈 TIME-BASED STATISTICS:")
    print("-" * 40)
    
    # Group samples by hour
    hourly_stats = {}
    for sample in all_samples:
        hour = sample['prediction_timestamp'][:13]  # YYYY-MM-DD_HH
        hourly_stats[hour] = hourly_stats.get(hour, 0) + 1
    
    for hour in sorted(hourly_stats.keys()):
        print(f"  {hour}: {hourly_stats[hour]} samples")
    
    return result

def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "tiff_files"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "training_sequences.json"
    
    if not os.path.exists(directory):
        print(f"❌ Directory {directory} does not exist")
        sys.exit(1)
    
    analyze_tiff_training_data(directory, output_file)

if __name__ == "__main__":
    main()
