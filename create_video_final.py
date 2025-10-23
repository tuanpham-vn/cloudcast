#!/usr/bin/env python3
"""
Script tạo video từ ảnh PNG - phiên bản hoàn chỉnh
Hỗ trợ cả FFmpeg và OpenCV, xử lý đúng ảnh RGBA
"""

import os
import sys
import glob
import subprocess
import cv2
import numpy as np
from PIL import Image
import re
from datetime import datetime


def sort_by_timestamp(image_files):
    """Sắp xếp ảnh theo timestamp thực tế từ cũ đến mới"""

    def extract_timestamp(filename):
        # Extract timestamp từ filename: radar_YYYY-MM-DD_HHMM_pred.png (format cũ)
        match = re.search(r'radar_(\d{4}-\d{2}-\d{2}_\d{4})_pred\.png', filename)
        if match:
            timestamp_str = match.group(1)
            return datetime.strptime(timestamp_str, '%Y-%m-%d_%H%M')

        # Extract timestamp từ filename: YYYY-MM-DD_HHMM_acc10.png (format mới)
        match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{4})_acc\d+\.png', filename)
        if match:
            timestamp_str = match.group(1)
            return datetime.strptime(timestamp_str, '%Y-%m-%d_%H%M')

        return datetime.min  # Nếu không match, đặt ở đầu

    return sorted(image_files, key=extract_timestamp)


def create_video_with_ffmpeg(input_dir="predictions/png", output_file="video_ffmpeg.mp4", fps=4):
    """Tạo video sử dụng FFmpeg với xử lý kích thước ảnh"""

    # Tìm ảnh PNG và sắp xếp theo timestamp
    all_files = glob.glob(os.path.join(input_dir, "*.png"))
    image_files = sort_by_timestamp(all_files)

    if not image_files:
        print(f"ERROR: Không tìm thấy ảnh PNG trong {input_dir}")
        return False

    print(f"Tìm thấy {len(image_files)} ảnh PNG")

    # Phân loại file theo pattern
    radar_files = [f for f in image_files if 'radar_' in f and '_pred.png' in f]
    acc_files = [f for f in image_files if re.search(r'\d{4}-\d{2}-\d{2}_\d{4}_acc\d+\.png', f)]

    if radar_files:
        print(f"  - {len(radar_files)} file radar_*_pred.png (format cũ)")
    if acc_files:
        print(f"  - {len(acc_files)} file YYYY-MM-DD_HHMM_acc*.png (format mới)")

    print(f"Tạo video {fps} FPS...")

    # Kiểm tra ảnh đầu tiên
    first_img = Image.open(image_files[0])
    width, height = first_img.size
    print(f"Kích thước ảnh: {width}x{height}")
    print(f"Mode ảnh: {first_img.mode}")

    # Đảm bảo kích thước chia hết cho 2 (yêu cầu của H264)
    if width % 2 != 0:
        width = width - 1
    if height % 2 != 0:
        height = height - 1

    print(f"Kích thước sau điều chỉnh: {width}x{height}")

    # Tạo danh sách file ảnh đã sắp xếp cho FFmpeg
    # FFmpeg cần file list thay vì pattern glob cho nhiều loại file
    file_list_path = os.path.join(input_dir, "file_list.txt")
    with open(file_list_path, 'w') as f:
        for img_file in image_files:
            f.write(f"file '{os.path.basename(img_file)}'\n")
            f.write(f"duration {1.0 / fps}\n")
        # Thêm frame cuối cùng một lần nữa để đảm bảo hiển thị đủ thời gian
        if image_files:
            f.write(f"file '{os.path.basename(image_files[-1])}'\n")

    # Lệnh FFmpeg với file list
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-f', 'concat',
        '-safe', '0',
        '-i', file_list_path,
        '-vf', f'scale={width}:{height}',  # Scale để đảm bảo kích thước chia hết cho 2
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',  # High quality
        output_file
    ]

    print(f"Chạy lệnh: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ FFmpeg thành công!")
        if result.stderr:
            print(f"FFmpeg info: {result.stderr.split('Stream mapping:')[0].strip()}")

        # Cleanup file list
        if os.path.exists(file_list_path):
            os.remove(file_list_path)

        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg thất bại: {e}")
        if e.stderr:
            print(f"Lỗi: {e.stderr.split('Conversion failed!')[0].strip()}")

        # Cleanup file list
        if os.path.exists(file_list_path):
            os.remove(file_list_path)

        return False
    except FileNotFoundError:
        print("❌ FFmpeg không được cài đặt")
        return False


def create_video_with_opencv(input_dir="predictions/png", output_file="video_opencv.mp4", fps=4):
    """Tạo video với OpenCV - xử lý RGBA đúng cách"""

    # Tìm ảnh PNG và sắp xếp theo timestamp
    all_files = glob.glob(os.path.join(input_dir, "*.png"))
    image_files = sort_by_timestamp(all_files)

    if not image_files:
        print(f"ERROR: Không tìm thấy ảnh PNG trong {input_dir}")
        return False

    print(f"Tìm thấy {len(image_files)} ảnh PNG")

    # Phân loại file theo pattern
    radar_files = [f for f in image_files if 'radar_' in f and '_pred.png' in f]
    acc_files = [f for f in image_files if re.search(r'\d{4}-\d{2}-\d{2}_\d{4}_acc\d+\.png', f)]

    if radar_files:
        print(f"  - {len(radar_files)} file radar_*_pred.png (format cũ)")
    if acc_files:
        print(f"  - {len(acc_files)} file YYYY-MM-DD_HHMM_acc*.png (format mới)")

    # Đọc ảnh đầu tiên để lấy kích thước
    first_img = Image.open(image_files[0])
    if first_img.mode == 'RGBA':
        background = Image.new('RGBA', first_img.size, (255, 255, 255, 255))
        first_img = Image.alpha_composite(background, first_img)
        first_img = first_img.convert('RGB')

    width, height = first_img.size
    print(f"Kích thước: {width}x{height}")

    # Thử codec XVID với container AVI
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_file_avi = output_file.replace('.mp4', '.avi')
    video = cv2.VideoWriter(output_file_avi, fourcc, fps, (width, height))

    if not video.isOpened():
        print("❌ Không thể tạo video writer")
        return False

    print(f"✓ Sử dụng codec XVID, file: {output_file_avi}")

    success_count = 0
    for i, img_path in enumerate(image_files):
        # Đọc và xử lý ảnh RGBA
        pil_img = Image.open(img_path)

        if pil_img.mode == 'RGBA':
            background = Image.new('RGBA', pil_img.size, (255, 255, 255, 255))
            pil_img = Image.alpha_composite(background, pil_img)
            pil_img = pil_img.convert('RGB')
        elif pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        # Chuyển sang numpy và BGR
        img_array = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Resize nếu cần
        if img_bgr.shape[:2] != (height, width):
            img_bgr = cv2.resize(img_bgr, (width, height))

        # Ghi frame
        video.write(img_bgr)
        success_count += 1

        if (i + 1) % 5 == 0:
            print(f"Đã xử lý {i + 1}/{len(image_files)} ảnh...")

    video.release()

    # Kiểm tra file AVI
    if os.path.exists(output_file_avi) and os.path.getsize(output_file_avi) > 0:
        print(f"✅ Video AVI đã tạo: {output_file_avi}")
        print(f"   Kích thước: {os.path.getsize(output_file_avi)} bytes")
        print(f"   Số ảnh: {success_count}/{len(image_files)}")

        # Thử chuyển đổi sang MP4 nếu có FFmpeg
        try:
            cmd = ['ffmpeg', '-y', '-i', output_file_avi, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_file]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ Đã chuyển đổi sang MP4: {output_file}")
            os.remove(output_file_avi)  # Xóa file AVI tạm
            return True
        except:
            print(f"⚠️  Không thể chuyển sang MP4, giữ file AVI: {output_file_avi}")
            return True
    else:
        print("❌ Video không được tạo")
        return False


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "predictions/png"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "predictions/png/prediction.mp4"
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    print("=" * 60)
    print("TẠO VIDEO TỪ ẢNH PNG - PHIÊN BẢN HOÀN CHỈNH")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_file}")
    print(f"FPS:    {fps}")
    print("=" * 60)

    # Thử FFmpeg trước (chất lượng cao hơn)
    print("\n🎬 Thử FFmpeg...")
    success = create_video_with_ffmpeg(input_dir, output_file, fps)

    if not success:
        print("\n🎬 Thử OpenCV...")
        success = create_video_with_opencv(input_dir, output_file, fps)

    if success:
        print("\n🎉 HOÀN THÀNH!")
        print(f"Video đã được tạo: {output_file}")

        # Kiểm tra file cuối cùng
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"Kích thước file: {file_size:,} bytes")

            # Test đọc video
            cap = cv2.VideoCapture(output_file)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Video có thể phát được (kích thước frame: {frame.shape})")
                cap.release()
    else:
        print("\n❌ THẤT BẠI!")
        print("Không thể tạo video với bất kỳ phương pháp nào")
        sys.exit(1)


if __name__ == "__main__":
    main()
