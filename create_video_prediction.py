#!/usr/bin/env python3
"""
Script đơn giản tạo video từ ảnh PNG - chỉ ghép ảnh lại.
"""

import os
import sys
import glob
import cv2
import numpy as np

def create_simple_video(input_dir="predictions/png", output_file="simple_video.mp4", fps=4):
    """Tạo video đơn giản từ ảnh PNG."""
    
    # Tìm ảnh PNG
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    
    if not image_files:
        print(f"ERROR: Không tìm thấy ảnh PNG trong {input_dir}")
        return False
    
    print(f"Tìm thấy {len(image_files)} ảnh PNG")
    print(f"Tạo video {fps} FPS...")
    
    # Đọc ảnh đầu tiên để lấy kích thước
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print("ERROR: Không thể đọc ảnh đầu tiên")
        return False
    
    height, width = first_img.shape[:2]
    print(f"Kích thước: {width}x{height}")
    
    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Ghép từng ảnh
    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is not None:
            # Resize nếu cần
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            video.write(img)
        
        if (i + 1) % 5 == 0:
            print(f"Đã xử lý {i + 1}/{len(image_files)} ảnh...")
    
    video.release()
    print(f"✅ Video đã tạo: {output_file}")
    return True

if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "predictions/png"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "simple_video.mp4"
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    
    create_simple_video(input_dir, output_file, fps)


'''
python create_video_prediction.py  predictions/png radar_prediction.mp4 4
'''