import numpy as np
import glob
import sys
import datetime
import os

DATA_DIR = os.environ.get("CLOUDCAST_DATA_DIR", "./data")

def load_npy_file(file_path):
    """
    Đọc dữ liệu từ tệp .npz
    
    Args:
        file_path: Đường dẫn đến tệp .npz
        
    Returns:
        Dữ liệu từ tệp .npz
    """
    try:
        return np.load(file_path)
    except Exception as e:
        print(f"Lỗi khi đọc tệp {file_path}: {e}")
        return None

def get_npy_files(directory, pattern="*.npz"):
    """
    Lấy danh sách các tệp .npz trong thư mục
    
    Args:
        directory: Thư mục chứa tệp .npz
        pattern: Mẫu tên tệp để tìm kiếm
        
    Returns:
        Danh sách các đường dẫn tệp .npz
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    return files
