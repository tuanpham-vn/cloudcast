#!/usr/bin/env python3
"""
Script phân tích timestamp của các file TIF để tìm chuỗi liên tiếp phù hợp cho training
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

def find_continuous_sequences(timestamps, max_gap=2):
    """
    Tìm các chuỗi liên tiếp với gap tối đa là max_gap
    Trả về list các chuỗi, mỗi chuỗi là (start_idx, end_idx, missing_count)
    """
    if not timestamps:
        return []
    
    sequences = []
    current_start = 0
    missing_count = 0
    
    for i in range(1, len(timestamps)):
        current_time = parse_timestamp(timestamps[i-1])
        next_time = parse_timestamp(timestamps[i])
        expected_next = current_time + timedelta(minutes=10)
        
        if next_time == expected_next:
            # Liên tiếp hoàn hảo
            continue
        elif next_time > expected_next:
            # Có gap
            gap_minutes = (next_time - expected_next).total_seconds() / 60
            gap_count = int(gap_minutes / 10)  # Số file bị thiếu
            
            if gap_count <= max_gap:
                missing_count += gap_count
            else:
                # Gap quá lớn, kết thúc chuỗi hiện tại
                if i - current_start >= 1:  # Chuỗi có ít nhất 2 phần tử
                    sequences.append((current_start, i-1, missing_count))
                current_start = i
                missing_count = 0
    
    # Thêm chuỗi cuối cùng
    if len(timestamps) - current_start >= 1:
        sequences.append((current_start, len(timestamps)-1, missing_count))
    
    return sequences

def analyze_tiff_directory(directory="/media/databourg/9064c7cc-2acd-4a8a-9d9e-9212e26e08b92/Himawari_Rain_result_10mins_grids_5Mb"):
    """Phân tích thư mục chứa file TIF"""
    
    # Tìm tất cả file TIF
    pattern = os.path.join(directory, "*.tif")
    tif_files = glob.glob(pattern)
    
    if not tif_files:
        print(f"❌ Không tìm thấy file TIF nào trong {directory}")
        return
    
    print(f"📁 Phân tích thư mục: {directory}")
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
    print(f"✅ Tìm thấy {len(timestamps)} file có timestamp hợp lệ")
    
    if not timestamps:
        print("❌ Không có timestamp hợp lệ nào")
        return
    
    # Parse timestamps thành datetime objects
    dt_timestamps = [parse_timestamp(ts) for ts in timestamps]
    start_time = min(dt_timestamps)
    end_time = max(dt_timestamps)
    
    print(f"📅 Khoảng thời gian: {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Tạo danh sách timestamp mong đợi
    expected_timestamps = generate_expected_timestamps(start_time, end_time)
    expected_set = set(expected_timestamps)
    actual_set = set(timestamps)
    
    # Tìm missing timestamps
    missing_timestamps = expected_set - actual_set
    missing_count = len(missing_timestamps)
    
    print(f"❌ Số timestamp bị thiếu: {missing_count}")
    print(f"📈 Tỷ lệ hoàn chỉnh: {((len(expected_timestamps) - missing_count) / len(expected_timestamps) * 100):.1f}%")
    
    # Phân tích gaps
    print("\n" + "="*80)
    print("🔍 PHÂN TÍCH GAPS")
    print("="*80)
    
    gap_stats = defaultdict(int)
    current_gap = 0
    
    for i in range(1, len(dt_timestamps)):
        prev_time = dt_timestamps[i-1]
        curr_time = dt_timestamps[i]
        expected_time = prev_time + timedelta(minutes=10)
        
        if curr_time > expected_time:
            gap_minutes = (curr_time - expected_time).total_seconds() / 60
            gap_count = int(gap_minutes / 10)
            gap_stats[gap_count] += 1
            current_gap = max(current_gap, gap_count)
    
    print("Thống kê gaps:")
    for gap_size in sorted(gap_stats.keys()):
        count = gap_stats[gap_size]
        print(f"  - Gap {gap_size} file: {count} lần")
    
    print(f"Gap lớn nhất: {current_gap} file")
    
    # Tìm chuỗi liên tiếp với gap <= 2
    print("\n" + "="*80)
    print("🔗 CÁC CHUỖI LIÊN TIẾP (Gap ≤ 2)")
    print("="*80)
    
    sequences = find_continuous_sequences(timestamps, max_gap=2)
    
    if not sequences:
        print("❌ Không tìm thấy chuỗi nào với gap ≤ 2")
        return
    
    # Sắp xếp theo độ dài chuỗi (giảm dần)
    sequences.sort(key=lambda x: x[1] - x[0] + 1, reverse=True)
    
    print(f"✅ Tìm thấy {len(sequences)} chuỗi liên tiếp")
    print()
    
    for i, (start_idx, end_idx, missing_count) in enumerate(sequences, 1):
        length = end_idx - start_idx + 1
        start_ts = timestamps[start_idx]
        end_ts = timestamps[end_idx]
        
        print(f"Chuỗi {i}:")
        print(f"  📏 Độ dài: {length} file")
        print(f"  ⏰ Thời gian: {start_ts} → {end_ts}")
        print(f"  ❌ Thiếu: {missing_count} file")
        print(f"  📈 Tỷ lệ hoàn chỉnh: {((length - missing_count) / length * 100):.1f}%")
        
        # Hiển thị một số file đầu và cuối
        if length <= 10:
            print(f"  📁 Files: {', '.join(timestamps[start_idx:end_idx+1])}")
        else:
            print(f"  📁 Files: {timestamps[start_idx]} ... {timestamps[end_idx]} ({length} files)")
        print()
    
    # Thống kê tổng quan
    print("="*80)
    print("📊 THỐNG KÊ TỔNG QUAN")
    print("="*80)
    
    total_files_in_sequences = sum(end_idx - start_idx + 1 for _, end_idx, _ in sequences)
    total_missing_in_sequences = sum(missing_count for _, _, missing_count in sequences)
    
    print(f"Tổng file trong các chuỗi: {total_files_in_sequences}")
    print(f"Tổng file bị thiếu trong chuỗi: {total_missing_in_sequences}")
    print(f"Tỷ lệ hoàn chỉnh trung bình: {((total_files_in_sequences - total_missing_in_sequences) / total_files_in_sequences * 100):.1f}%")
    
    # Top 5 chuỗi dài nhất
    print(f"\n🏆 TOP 5 CHUỖI DÀI NHẤT:")
    for i, (start_idx, end_idx, missing_count) in enumerate(sequences[:5], 1):
        length = end_idx - start_idx + 1
        start_ts = timestamps[start_idx]
        end_ts = timestamps[end_idx]
        print(f"  {i}. {length} files ({start_ts} → {end_ts}) - Thiếu {missing_count} files")

def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "tiff_files"
    
    if not os.path.exists(directory):
        print(f"❌ Thư mục {directory} không tồn tại")
        sys.exit(1)
    
    analyze_tiff_directory(directory)

if __name__ == "__main__":
    main()
