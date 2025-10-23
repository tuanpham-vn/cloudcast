import numpy as np
import sys
import cv2
from scipy import ndimage


def get_img_size(preprocess):
    for x in preprocess.split(","):
        k, v = x.split("=")
        if k == "img_size":
            return tuple(map(int, v.split("x")))
    return None


# Chỉ giữ lại các hàm cần thiết cho xử lý dữ liệu ảnh mây và leadtime_conditioning


def to_binary_mask(arr):
    arr[arr < 0.1] = 0.0
    arr[arr > 0] = 1.0

    return arr


def to_classes(arr, num_classes):
    return (np.around((100.0 * arr) / num_classes, decimals=0) * num_classes) / 100.0


def preprocess_many(imgs, process_label):
    return np.asarray(list(map(lambda x: preprocess_single(x, process_label), imgs)))


def preprocess_single(arr, process_label):
    for proc in process_label.split(","):
        k, v = proc.split("=")
        if k == "conv":
            v = int(v)
            kern = np.ones((v, v), np.float32) / (v * v)
            kern = np.expand_dims(kern, axis=2)
            arr = ndimage.convolve(arr, kern, mode="constant", cval=0.0)
        elif k == "to_binary_mask" and v == "true":
            arr = to_binary_mask(arr)
        elif k == "classes":
            arr = to_classes(arr, int(v))
        elif k == "standardize" and v == "true":
            arr = (arr - arr.mean()) / arr.std()
        elif k == "normalize" and v == "true":
            if np.min(arr) != np.max(arr):
                arr = (arr - np.min(arr)) / np.ptp(arr)
        elif k == "img_size":
            img_size = tuple(map(int, v.split("x")))
            if arr.shape != img_size:
                arr = np.expand_dims(
                    cv2.resize(arr, dsize=img_size, interpolation=cv2.INTER_LINEAR),
                    axis=2,
                )
        elif k == "area":  # Không còn xử lý area
            pass

    return arr


# Các hàm liên quan đến thời gian, góc mặt trời, địa hình, v.v. đã được loại bỏ
# vì không cần thiết cho việc xử lý dữ liệu ảnh mây và leadtime_conditioning


def create_onehot_leadtime_conditioning(img_size, depth, active_layer):
    b = np.ones((1,) + img_size)
    return np.expand_dims(
        np.expand_dims(np.expand_dims(np.eye(depth)[active_layer], -1), 1) * b, axis=-1
    ).astype(np.short)


def create_squeezed_leadtime_conditioning(img_size, depth, active_leadtime):
    return np.expand_dims(
        np.full(img_size, active_leadtime / depth), axis=(0, 3)
    ).astype(np.float32)


# Các hàm liên quan đến địa hình, loại địa hình, và khí hậu đã được loại bỏ
# vì không cần thiết cho việc xử lý dữ liệu ảnh mây và leadtime_conditioning
