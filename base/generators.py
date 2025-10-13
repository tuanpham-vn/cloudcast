import datetime
import numpy as np
from tensorflow import keras
from base.fileutils import load_npy_file, get_npy_files
from base.preprocess import preprocess_single


# datetime ring buffer
#
# datetimes are generated so that forecast analysis time fits between the given range
#
# for example
# * history length = 2
# * prediction length = 2
# * single time = 202111T0800
#
# times are:
# - 20211107T2330
# - 20211107T2345
# - 20211108T0000
# - 20211108T0015


class TimeseriesGenerator:
    def __init__(
        self,
        start_date,
        stop_date,
        history_len,
        pred_len,
        # Sử dụng khoảng thời gian 10 phút thay vì 15 phút
        step=datetime.timedelta(minutes=10),
    ):
        self.date = start_date
        self.stop_date = stop_date
        self.history_len = history_len
        self.prediction_len = pred_len
        self.step = step
        self.times = [start_date - (history_len - 1) * step]
        self.create()
        assert start_date is not None and stop_date is not None

    def __iter__(self):
        while True:
            yield self.times
            self.create()
            if self.times[-1] > self.stop_date:
                break

    def create(self):
        if len(self.times) > 1:
            self.times.pop(0)
        while len(self.times) < self.history_len + self.prediction_len:
            self.times.append(self.times[-1] + self.step)


class DataSeries:
    def __init__(
        self,
        data_dir,
        preprocess=None,
        cache_data=True,
    ):
        """        
        Lớp xử lý dữ liệu cho các tệp .npz
        
        Args:
            data_dir: Thư mục chứa dữ liệu .npz
            preprocess: Chuỗi tiền xử lý
            cache_data: Có lưu trữ dữ liệu trong bộ nhớ hay không
        """
        self.data_series = {}
        self.data_dir = data_dir
        self.preprocess = preprocess
        self.cache_data = cache_data

    def read_data(self, times):
        """
        Đọc dữ liệu từ các tệp .npz theo thời gian
        
        Args:
            times: Danh sách các thời gian cần đọc
            
        Returns:
            Mảng dữ liệu đọc được
        """
        datakeys = list(self.data_series.keys())
        new_series = {}

        if self.cache_data:
            new_times = list(set(datakeys + times))
        else:
            new_times = times

        new_times.sort()

        for t in new_times:
            if t in datakeys:
                new_series[t] = self.data_series[t]
            else:
                # Đọc dữ liệu từ tệp .npz
                file_path = f"{self.data_dir}/{t}.npz"
                data_loaded = load_npy_file(file_path)
                data = data_loaded["data"] if data_loaded is not None else None
                
                if data is not None:
                    if self.preprocess:
                        data = preprocess_single(data, self.preprocess)
                    new_series[t] = data
                else:
                    # Tạo dữ liệu trống nếu không tìm thấy tệp
                    print(f"Không tìm thấy tệp {file_path}, tạo dữ liệu trống")
                    new_series[t] = np.zeros((512, 512, 1), dtype=np.float32)

        self.data_series = new_series

        return np.asarray(list(self.data_series.values()))
