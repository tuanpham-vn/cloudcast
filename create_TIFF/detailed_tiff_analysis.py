#!/usr/bin/env python3
"""
Script phân tích chi tiết các file TIF bị thiếu và đề xuất chuỗi training
"""

import os
import glob
import re
from datetime import datetime, timedelta
from collections import defaultdict
import sys

def extract_timestamp_from_filename(filename):
    """Extract timestamp từ filename YYYY-MM-DD_HHMM_acc10.tif"""
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{4})_acc\d+\.tif', filename)
    if match:
        return match.group(1)
    return None

def parse_timestamp(ts_str):
    """Parse timestamp string thành datetime object"""
    return datetime.strptime(ts_str, '%Y-%m-%d_%H%M')

def generate_expected_timestamps(start_time, end_time, interval_minutes=10):
    """Tạo danh sách timestamp mong đợi từ start_time đến end_time"""
    expected = []
    current = start_time
    while current <= end_time:
        expected.append(current.strftime('%Y-%m-%d_%H%M'))
        current += timedelta(minutes=interval_minutes)
    return expected

def find_missing_timestamps(expected, actual):
    """Tìm các timestamp bị thiếu"""
    expected_set = set(expected)
    actual_set = set(actual)
    missing = sorted(list(expected_set - actual_set))
    return missing

def find_gaps_in_sequence(timestamps):
    """Tìm các gap trong chuỗi timestamp"""
    gaps = []
    for i in range(1, len(timestamps)):
        prev_time = parse_timestamp(timestamps[i-1])
        curr_time = parse_timestamp(timestamps[i])
        expected_time = prev_time + timedelta(minutes=10)
        
        if curr_time > expected_time:
            gap_minutes = (curr_time - expected_time).total_seconds() / 60
            gap_count = int(gap_minutes / 10)
            
            # Tìm các timestamp bị thiếu trong gap này
            missing_in_gap = []
            temp_time = expected_time
            while temp_time < curr_time:
                missing_in_gap.append(temp_time.strftime('%Y-%m-%d_%H%M'))
                temp_time += timedelta(minutes=10)
            
            gaps.append({
                'position': i,
                'prev_timestamp': timestamps[i-1],
                'curr_timestamp': timestamps[i],
                'gap_count': gap_count,
                'missing_timestamps': missing_in_gap
            })
    
    return gaps

def suggest_training_sequences(timestamps, min_sequence_length=20, max_gap=2):
    """Đề xuất các chuỗi phù hợp cho training"""
    sequences = []
    current_start = 0
    current_missing = 0
    
    for i in range(1, len(timestamps)):
        prev_time = parse_timestamp(timestamps[i-1])
        curr_time = parse_timestamp(timestamps[i])
        expected_time = prev_time + timedelta(minutes=10)
        
        if curr_time == expected_time:
            # Liên tiếp hoàn hảo
            continue
        elif curr_time > expected_time:
            # Có gap
            gap_minutes = (curr_time - expected_time).total_seconds() / 60
            gap_count = int(gap_minutes / 10)
            
            if gap_count <= max_gap:
                current_missing += gap_count
            else:
                # Gap quá lớn, kết thúc chuỗi hiện tại
                if i - current_start >= min_sequence_length:
                    sequences.append({
                        'start_idx': current_start,
                        'end_idx': i-1,
                        'length': i - current_start,
                        'missing_count': current_missing,
                        'start_time': timestamps[current_start],
                        'end_time': timestamps[i-1]
                    })
                current_start = i
                current_missing = 0
    
    # Thêm chuỗi cuối cùng
    if len(timestamps) - current_start >= min_sequence_length:
        sequences.append({
            'start_idx': current_start,
            'end_idx': len(timestamps)-1,
            'length': len(timestamps) - current_start,
            'missing_count': current_missing,
            'start_time': timestamps[current_start],
            'end_time': timestamps[len(timestamps)-1]
        })
    
    return sequences

def analyze_tiff_directory_detailed(directory="tiff_files"):
    """Phân tích chi tiết thư mục chứa file TIF"""
    
    # Tìm tất cả file TIF
    pattern = os.path.join(directory, "*.tif")
    tif_files = glob.glob(pattern)
    
    if not tif_files:
        print(f"❌ Không tìm thấy file TIF nào trong {directory}")
        return
    
    print(f"📁 Phân tích chi tiết thư mục: {directory}")
    print(f"📊 Tổng số file TIF: {len(tif_files)}")
    print("="*80)
    
    # Extract timestamps
    timestamps = []
    for file_path in tif_files:
        filename = os.path.basename(file_path)
        ts = extract_timestamp_from_filename(filename)
        if ts:
            timestamps.append(ts)
    
    timestamps.sort()
    
    # Parse timestamps thành datetime objects
    dt_timestamps = [parse_timestamp(ts) for ts in timestamps]
    start_time = min(dt_timestamps)
    end_time = max(dt_timestamps)
    
    # Tạo danh sách timestamp mong đợi
    expected_timestamps = generate_expected_timestamps(start_time, end_time)
    
    # Tìm missing timestamps
    missing_timestamps = find_missing_timestamps(expected_timestamps, timestamps)
    
    print(f"📅 Khoảng thời gian: {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"📊 Tổng timestamp mong đợi: {len(expected_timestamps)}")
    print(f"✅ Timestamp có sẵn: {len(timestamps)}")
    print(f"❌ Timestamp bị thiếu: {len(missing_timestamps)}")
    print(f"📈 Tỷ lệ hoàn chỉnh: {((len(timestamps) / len(expected_timestamps)) * 100):.1f}%")
    
    # Hiển thị các file bị thiếu
    if missing_timestamps:
        print(f"\n🔍 CÁC FILE BỊ THIẾU:")
        print("-" * 50)
        for i, missing_ts in enumerate(missing_timestamps, 1):
            print(f"  {i}. {missing_ts}_acc10.tif")
    
    # Phân tích gaps
    print(f"\n🔍 PHÂN TÍCH GAPS TRONG CHUỖI:")
    print("-" * 50)
    gaps = find_gaps_in_sequence(timestamps)
    
    if gaps:
        for i, gap in enumerate(gaps, 1):
            print(f"Gap {i}:")
            print(f"  Vị trí: Sau {gap['prev_timestamp']}")
            print(f"  Gap: {gap['gap_count']} file")
            print(f"  Thiếu: {', '.join(gap['missing_timestamps'])}")
            print()
    else:
        print("✅ Không có gap nào trong chuỗi")
    
    # Đề xuất chuỗi training
    print("="*80)
    print("🎯 ĐỀ XUẤT CHUỖI CHO TRAINING")
    print("="*80)
    
    # Thử các độ dài chuỗi tối thiểu khác nhau
    for min_length in [10, 20, 50, 100]:
        sequences = suggest_training_sequences(timestamps, min_length, max_gap=2)
        if sequences:
            print(f"\n📏 Chuỗi có độ dài ≥ {min_length} file:")
            print("-" * 40)
            for i, seq in enumerate(sequences, 1):
                completeness = ((seq['length'] - seq['missing_count']) / seq['length'] * 100)
                print(f"  {i}. {seq['length']} files ({seq['start_time']} → {seq['end_time']})")
                print(f"     Thiếu: {seq['missing_count']} files, Hoàn chỉnh: {completeness:.1f}%")
        else:
            print(f"\n❌ Không có chuỗi nào dài ≥ {min_length} file")
    
    # Phân tích cho machine learning
    print("\n" + "="*80)
    print("🤖 PHÂN TÍCH CHO MACHINE LEARNING")
    print("="*80)
    
    # Giả sử cần ít nhất 4 frame history + 1 prediction
    min_ml_sequence = 5
    ml_sequences = suggest_training_sequences(timestamps, min_ml_sequence, max_gap=2)
    
    if ml_sequences:
        print(f"✅ Tìm thấy {len(ml_sequences)} chuỗi phù hợp cho ML (≥{min_ml_sequence} frames)")
        
        total_samples = 0
        for seq in ml_sequences:
            # Số sample có thể tạo từ chuỗi này
            # Với 4 frame history + 1 prediction, mỗi chuỗi dài n có thể tạo n-4 samples
            samples = max(0, seq['length'] - 4)
            total_samples += samples
            
            print(f"  - Chuỗi {seq['length']} frames → {samples} training samples")
        
        print(f"\n📊 Tổng training samples có thể tạo: {total_samples}")
    else:
        print(f"❌ Không có chuỗi nào phù hợp cho ML (cần ≥{min_ml_sequence} frames)")

def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "tiff_files"
    
    if not os.path.exists(directory):
        print(f"❌ Thư mục {directory} không tồn tại")
        sys.exit(1)
    
    analyze_tiff_directory_detailed(directory)

if __name__ == "__main__":
    main()
