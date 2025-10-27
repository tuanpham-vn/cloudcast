import tensorflow as tf
from tensorflow.data import AUTOTUNE
import numpy as np
import glob
import os
from datetime import datetime, timedelta
from enum import Enum
import copy
import sys
from base.fileutils import get_npy_files, DATA_DIR
from base.preprocess import (
    create_squeezed_leadtime_conditioning,
    get_img_size,
)


OpMode = Enum("OperatingMode", ["TRAIN", "INFER", "VERIFY"])


def normalize_time_string(ts_str):
    """Normalize timestamp strings to format YYYYMMDDTHHMMSS.

    Accepts inputs like 'YYYY-MM-DDTHHMM' and returns 'YYYYMMDDTHHMMSS'.
    """
    s = str(ts_str)
    if "T" in s:
        date_part, time_part = s.split("T", 1)
        date_part = date_part.replace("-", "")
        if len(time_part) == 4:
            time_part = time_part + "00"
        elif len(time_part) == 6:
            pass
        else:
            time_part = (time_part + "000000")[:6]
        return f"{date_part}T{time_part}"
    s = s.replace("-", "")
    if "T" not in s and len(s) >= 8:
        return s
    return s


def read_times_from_preformatted_files_directory(dirname):
    """
    Read times from directory containing preformatted files.
    Supports two formats:
    1. Old format: *-times.npy files (one time series)
    2. New format: *.npz files with arr_0 (data) and arr_1 (times) - multiple patches
    Returns times (list), toc (dict), data_cache (dict)
    """
    toc = {}
    data_cache = {}

    time_files = glob.glob(f"{dirname}/*-times.npy")
    if time_files:
        for f in time_files:
            times = np.load(f)
            for i, t in enumerate(times):
                toc[t] = {"filename": f.replace("-times", ""), "index": i, "time": t}
        times = list(toc.keys())
        times.sort()
        print("Read {} times from {} (old format)".format(len(times), dirname))
        return times, toc, data_cache

    npz_files = sorted(glob.glob(f"{dirname}/*.npz"))
    if not npz_files:
        print(f"ERROR: No NPZ files found in {dirname}")
        return [], {}, data_cache

    print("Found {} NPZ files in directory (new format - multiple patches)".format(len(npz_files)))
    file_idx = 0
    for npz_file in npz_files:
        try:
            ds = np.load(npz_file)
            data = ds["arr_0"]  # Data already clipped to [0, 100] in create_tiff_dataset.py
            times = ds["arr_1"]
            data_cache[npz_file] = data
            for i, t in enumerate(times):
                t_normalized = normalize_time_string(t)
                key = (t_normalized, file_idx)
                toc[key] = {
                    "filename": npz_file,
                    "index": i,
                    "time": t_normalized,
                    "file_idx": file_idx,
                }
            file_idx += 1
            print("  Loaded {} timesteps from {} (cached in memory)".format(len(times), os.path.basename(npz_file)))
        except Exception as e:
            print("  Warning: Failed to load {}: {}".format(npz_file, e))
            continue

    times = sorted(list(toc.keys()), key=lambda x: (x[0], x[1]))
    print("Total samples across all patches: {}".format(len(times)))
    print("Unique timestamps: {}".format(len(set([t[0] for t in times]))))
    print("Number of patches: {}".format(file_idx))
    return times, toc, data_cache


def read_datas_from_preformatted_files_directory(dirname, toc, times, data_cache=None):
    """Read data for given times from directory-based TOC, using cache when available."""
    datas = []
    for t in times:
        e = toc[t]
        idx = e["index"]
        filename = e["filename"]
        if filename.endswith('.npz'):
            if data_cache is not None and filename in data_cache:
                arr = data_cache[filename][idx]
            else:
                datafile = np.load(filename, mmap_mode="r")
                arr = datafile["arr_0"][idx]
        else:
            datafile = np.load(filename, mmap_mode="r")
            arr = datafile[idx]  # Data already clipped to [0, 100] in create_tiff_dataset.py
        datas.append(arr)
    # Return timestamps as strings
    times_str = []
    for t in times:
        if isinstance(t, tuple):
            times_str.append(t[0])
        else:
            times_str.append(t)
    return datas, times_str


def read_times_from_preformatted_file(filename):
    ds = np.load(filename)
    data = ds["arr_0"]  # Data already clipped to [0, 100] in create_tiff_dataset.py
    times = ds["arr_1"]
    toc = {}
    for i, t in enumerate(times):
        toc[t] = {"index": i, "time": t}
    return times, data, toc


def read_datas_from_preformatted_file(all_times, all_data, req_times, toc):
    datas = []
    for t in req_times:
        index = toc[t]["index"]
        datas.append(all_data[index])

    return datas, req_times


class DataSeriesGenerator:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        print(
            "Generator number of batches: {} batch size: {}".format(
                len(self), self.batch_size
            )
        )

    def __len__(self):
        """Return number of batches in this dataset"""
        return len(self.placeholder) // self.batch_size

    def __getitem__(self, idx):
        ph = self.placeholder[idx]

        X = ph[0]
        Y = ph[1]

        x_hist = X[0 : self.n_channels]

        x, y, xtimes, ytimes = self.get_xy(x_hist, [Y])

        lc = X[self.n_channels]
        lt = np.expand_dims(self.leadtimes[lc], axis=0)

        x = np.asarray(x)
        y = np.asarray(y)

        y = np.squeeze(y, axis=0)

        x = np.concatenate((x, lt), axis=0)

        # Compute y_time preferring ground-truth timestamp when present
        base_stride = int(getattr(self, "sequence_stride_minutes", 10) or 10)
        if isinstance(ytimes, (list, tuple)) and len(ytimes) > 0:
            y_last = normalize_time_string(ytimes[0])
            y_time = datetime.strptime(y_last, "%Y%m%dT%H%M%S")
        else:
            xt_last = normalize_time_string(xtimes[-1])
            ts = datetime.strptime(xt_last, "%Y%m%dT%H%M%S")
            y_time = ts + timedelta(minutes=(1 + lc) * base_stride)

        x = np.squeeze(np.swapaxes(x, 0, 3))

        if self.operating_mode in (OpMode.VERIFY, OpMode.INFER):
            xt_norm = list(map(normalize_time_string, xtimes))
            return (
                x,
                y,
                np.append(xt_norm, y_time.strftime("%Y%m%dT%H%M%S")),
            )
        else:
            return (x, y)

    def get_xy(self, x_elems, y_elems):
        xtimes = []
        ytimes = []

        if self.dataseries_file is not None:
            x, xtimes = read_datas_from_preformatted_file(
                self.elements, self.data, x_elems, self.toc
            )
            y, ytimes = read_datas_from_preformatted_file(
                self.elements, self.data, y_elems, self.toc
            )

        elif self.dataseries_directory is not None:
            x, xtimes = read_datas_from_preformatted_files_directory(
                self.dataseries_directory, self.toc, x_elems, self.data_cache
            )
            y, ytimes = read_datas_from_preformatted_files_directory(
                self.dataseries_directory, self.toc, y_elems, self.data_cache
            )

        else:
            x = []
            for elem in x_elems:
                file_path = os.path.join(DATA_DIR, elem + ".npz")
                data = np.load(file_path)["data"]
                x.append(data)
            
            y = []
            for elem in y_elems:
                file_path = os.path.join(DATA_DIR, elem + ".npz")
                data = np.load(file_path)["data"]
                y.append(data)
            
            return x, y, x_elems, y_elems

        return x, y, xtimes, ytimes

    def __call__(self):
        # For training with .repeat(), we need infinite loop but controlled by steps_per_epoch
        while True:
            for i in range(len(self.placeholder)):
                elem = self.__getitem__(i)
                yield elem
            
            # Shuffle data after each full pass through the dataset
            self.on_epoch_end()
            
            # For non-training modes, break after one pass
            if self.operating_mode != OpMode.TRAIN:
                break

    def on_epoch_end(self):
        if self.shuffle_data:
            np.random.shuffle(self.placeholder)


class LazyDataSeries:
    def __init__(self, **kwargs):
        try:
            opts = kwargs["opts"]
            self.n_channels = opts.n_channels
            self.leadtime_conditioning = int(
                kwargs.get("leadtime_conditioning", opts.leadtime_conditioning)
            )
            self.img_size = get_img_size(opts.preprocess)

        except KeyError:
            self.n_channels = int(kwargs.get("n_channels"))
            self.img_size = kwargs.get("img_size")
            self.leadtime_conditioning = int(kwargs.get("leadtime_conditioning"))

        self.batch_size = int(kwargs.get("batch_size", 1))
        self.global_batch_size = kwargs.get("global_batch_size", None)
        self.dataseries_file = kwargs.get("dataseries_file", None)
        self.dataseries_directory = kwargs.get("dataseries_directory", None)
        self.start_date = kwargs.get("start_date", None)
        self.stop_date = kwargs.get("stop_date", None)
        self.reuse_y_as_x = kwargs.get("reuse_y_as_x", False)
        self.shuffle_data = kwargs.get("shuffle_data", True)
        self.debug = kwargs.get("enable_debug", False)
        self.filenames = kwargs.get("filenames", None)
        self.analysis_time = kwargs.get("analysis_time", None)
        self.hourly_prediction = kwargs.get("hourly_prediction", False)
        operating_mode = kwargs.get("operating_mode", "TRAIN")
        
        # Add rotation angle parameter, default to 180 degrees
        self.rotation_angle = int(kwargs.get("rotation_angle", 180))
        if self.rotation_angle not in [0, 90, 180, 270]:
            print(f"Warning: Invalid rotation angle {self.rotation_angle}. Using default 0 degrees.")
            self.rotation_angle = 0 #180

        self.cache = kwargs.get("enable_cache", False)
        self.sequence_stride_minutes = kwargs.get("sequence_stride_minutes", None)
        self.sequence_offset_minutes = kwargs.get("sequence_offset_minutes", 0)

        if operating_mode == "TRAIN":
            self.operating_mode = OpMode.TRAIN
        elif operating_mode == "INFER":
            self.operating_mode = OpMode.INFER
        elif operating_mode == "VERIFY":
            self.operating_mode = OpMode.VERIFY
        else:
            print("Invalid operating mode: {}".format(operating_mode))
            sys.exit(1)

        if self.operating_mode == OpMode.INFER:
            self.shuffle_data = False
            self.batch_size = 1

        elif self.operating_mode == OpMode.VERIFY:
            self.shuffle_data = False
            self.batch_size = 1

        self._placeholder = []
        self.initialize()

    def initialize(self):
        if self.leadtime_conditioning > 0:
            leadtimes = np.asarray(
                [
                    create_squeezed_leadtime_conditioning(
                        self.img_size, self.leadtime_conditioning, x
                    )
                    for x in range(self.leadtime_conditioning)
                ]
            )
            self.leadtimes = np.squeeze(leadtimes, 1)

        if self.dataseries_file is not None:
            self.elements, self.data, self.toc = read_times_from_preformatted_file(
                self.dataseries_file
            )
            self.data_cache = None

        elif self.dataseries_directory is not None:
            self.elements, self.toc, self.data_cache = read_times_from_preformatted_files_directory(
                self.dataseries_directory
            )

        else:
            if self.filenames is not None:
                self.elements = self.filenames
            else:
                self.elements = get_npy_files(DATA_DIR, "*.npz")
                self.elements = [os.path.basename(f).replace('.npz', '') for f in self.elements]
            self.elements.sort()
            self.data_cache = None

        # Optional filtering by stride minutes and offset
        if self.sequence_stride_minutes is not None:
            stride = int(self.sequence_stride_minutes)
            offset = int(self.sequence_offset_minutes) % stride

            def _get_time_str(elem):
                return elem[0] if isinstance(elem, tuple) else elem

            filtered = []
            for elem in self.elements:
                t = normalize_time_string(_get_time_str(elem))
                try:
                    mm = int(t[11:13])
                except Exception:
                    filtered.append(elem)
                    continue
                if (mm % stride) == offset:
                    filtered.append(elem)
            removed = len(self.elements) - len(filtered)
            if removed > 0:
                print("Filtered {} elements by stride {} min with offset {} min".format(removed, stride, offset))
            self.elements = filtered

        step = 1 if self.reuse_y_as_x else self.n_channels + self.leadtime_conditioning
        n_fut = self.leadtime_conditioning if self.operating_mode != OpMode.INFER else 0

        has_file_index = len(self.elements) > 0 and isinstance(self.elements[0], tuple)

        if has_file_index:
            groups = {}
            for elem in self.elements:
                file_idx = elem[1]
                groups.setdefault(file_idx, []).append(elem)

            for file_idx, elems in groups.items():
                elems.sort(key=lambda t: t[0])

            for file_idx, elems in groups.items():
                if (len(elems) - (self.n_channels + n_fut)) < 0:
                    continue
                i = 0
                while i <= len(elems) - (self.n_channels + n_fut):
                    x = list(elems[i : i + self.n_channels])

                    last_time_str = x[-1][0]
                    if self.hourly_prediction and last_time_str[-4:] != "0000":
                        i += step
                        continue

                    for lt in range(self.leadtime_conditioning):
                        x_ = copy.deepcopy(x)
                        x_.append(lt)

                        if self.operating_mode == OpMode.INFER:
                            y = "nan"
                        else:
                            y = elems[i + self.n_channels + lt]

                        self._placeholder.append([x_, y])

                    i += step
        else:
            assert (
                len(self.elements) - (self.n_channels + n_fut)
            ) >= 0, "Too few data to make a prediction: {} (need at least {})".format(
                len(self.elements), self.n_channels + n_fut
            )

            i = 0
            while i <= len(self.elements) - (self.n_channels + n_fut):
                x = list(self.elements[i : i + self.n_channels])

                last_time_str = x[-1][0] if isinstance(x[-1], tuple) else x[-1]
                if self.hourly_prediction and last_time_str[-4:] != "0000":
                    i += step
                    continue

                for lt in range(self.leadtime_conditioning):
                    x_ = copy.deepcopy(x)
                    x_.append(lt)

                    if self.operating_mode == OpMode.INFER:
                        y = "nan"
                    else:
                        y = self.elements[i + self.n_channels + lt]

                    self._placeholder.append([x_, y])

                i += step

        assert len(self._placeholder) > 0, "Placeholder array is empty"

        # Stats
        if self.dataseries_directory is not None:
            unique_files = set()
            per_file_counts = {}
            if len(self.elements) > 0 and isinstance(self.elements[0], tuple):
                for ts, fidx in self.elements:
                    unique_files.add(fidx)
                for fidx in unique_files:
                    per_file_counts[fidx] = len({ts for ts, fx in self.elements if fx == fidx})
            else:
                per_file_counts[0] = len(set(self.elements))

            print("=" * 70)
            print("DATASET STATISTICS:")
            print("=" * 70)
            print("Number of patch files:        {}".format(len(unique_files) or 1))
            print("Total timeseries elements:    {}".format(len(self.elements)))
            print("Training samples created:     {}".format(len(self._placeholder)))
            if per_file_counts:
                avg = len(self._placeholder) // max(len(per_file_counts), 1)
                print("Samples per patch (approx):   {}".format(avg))
            print("=" * 70)
        else:
            print(
                "Placeholder timeseries length: {} number of samples: {}".format(
                    len(self.elements), len(self._placeholder)
                )
            )

        if self.shuffle_data:
            np.random.shuffle(self._placeholder)

    def __len__(self):
        """Return number of samples"""
        return len(self._placeholder)

    def get_dataset(self, take_ratio=None, skip_ratio=None):
        def rotate_image(image, angle):
            """Rotate image by angle degrees"""
            k = angle // 90  # Number of 90-degree rotations
            if k == 0:
                return image
            return tf.image.rot90(image, k=k)

        def normalize_and_rotate(x, y, t, n):
            # Data is already clipped to [0, 100] in create_tiff_dataset.py
            # Normalize to [0, 1] by multiplying by 0.01 for all operating modes
            x = tf.concat([0.01 * x[..., 0:n], x[..., n:]], axis=-1)
            y = y * 0.01
            
            # Clip to [0, 1] range to prevent any outliers
            x = tf.clip_by_value(x, 0.0, 1.0)
            y = tf.clip_by_value(y, 0.0, 1.0)
            
            # Apply rotation to both input and target images
            x_rotated = tf.concat([
                rotate_image(x[..., :n], self.rotation_angle),
                x[..., n:]
            ], axis=-1)
            y_rotated = rotate_image(y, self.rotation_angle)
            
            if t is not None:
                return (x_rotated, y_rotated, t)
            else:
                return (x_rotated, y_rotated)

        placeholder = None

        if take_ratio is not None:
            l = int(len(self._placeholder) * take_ratio)
            placeholder = self._placeholder[0:l]

        if skip_ratio is not None:
            l = int(len(self._placeholder) * skip_ratio)
            placeholder = self._placeholder[l:]

        if placeholder is None:
            placeholder = copy.deepcopy(self._placeholder)

        x_dim_len = self.n_channels
        x_dim_len += 1 if self.leadtime_conditioning > 0 else 0

        sig = (
            tf.TensorSpec(
                shape=self.img_size + (x_dim_len,), dtype=tf.float32, name="x"
            ),
            tf.TensorSpec(shape=self.img_size + (1,), dtype=tf.float32, name="y"),
        )

        if self.operating_mode in (OpMode.INFER, OpMode.VERIFY):
            sig += (
                tf.TensorSpec(
                    shape=(self.n_channels + 1,), dtype=tf.string, name="times"
                ),
            )

        gen = DataSeriesGenerator(placeholder=placeholder, **self.__dict__)
        dataset = tf.data.Dataset.from_generator(gen, output_signature=sig)

        # Apply normalization and rotation for all operating modes
        parallel_calls = 2 if self.global_batch_size is not None else AUTOTUNE
        
        if self.operating_mode == OpMode.TRAIN:
            dataset = dataset.map(
                lambda x, y: normalize_and_rotate(x, y, None, self.n_channels),
                num_parallel_calls=parallel_calls
            )
        else:
            dataset = dataset.map(
                lambda x, y, t: normalize_and_rotate(x, y, t, self.n_channels),
                num_parallel_calls=parallel_calls
            )
        
        # Determine the correct batch size for dataset batching
        if self.global_batch_size is not None:
            dataset_batch_size = self.global_batch_size
            print(f"✓ Multi-GPU dataset batching: using global_batch_size={dataset_batch_size}")
        else:
            dataset_batch_size = self.batch_size
            print(f"✓ Single GPU dataset batching: using batch_size={dataset_batch_size}")
        
        # Ensure we have enough samples for at least one complete batch
        if len(placeholder) < dataset_batch_size:
            print(f"Warning: Not enough samples ({len(placeholder)}) for batch size {dataset_batch_size}")
            print("Reducing batch size to match available samples...")
            dataset_batch_size = len(placeholder)
            if self.global_batch_size is not None:
                self.global_batch_size = dataset_batch_size
            self.batch_size = dataset_batch_size
        
        # Batch the dataset with the appropriate batch size
        dataset = dataset.batch(dataset_batch_size, drop_remainder=True)
        
        # Additional safety: ensure all batches have exactly the same size
        def ensure_batch_size(x, y):
            return x, y
        
        if self.global_batch_size is not None:
            dataset = dataset.map(ensure_batch_size, num_parallel_calls=2)
        else:
            dataset = dataset.map(ensure_batch_size, num_parallel_calls=AUTOTUNE)
        
        if self.cache and len(placeholder) < 10000:
            dataset = dataset.cache()
        
        if self.operating_mode == OpMode.TRAIN and self.global_batch_size is None:
            dataset = dataset.repeat()
        
        dataset = dataset.prefetch(AUTOTUNE)

        return dataset
