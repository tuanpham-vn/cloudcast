import os
# Reduce TF logs and disable XLA by default to avoid excessive warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ.pop('TF_XLA_FLAGS', None)

from datetime import datetime
from tensorflow.keras.models import save_model
from model import *
from base.preprocess import *
from base.fileutils import *
from base.generators import *
from base.opts import CloudCastOptions
from base.dataseries import LazyDataSeries
import math
import argparse
import json
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np

EPOCHS = 500


def print_loss_function_info():
    """
    In thông tin về các hàm loss có sẵn và cách sử dụng chúng
    """
    print("Các hàm loss có sẵn trong CloudCast:")
    print("  - MeanSquaredError: Hàm loss cơ bản, tính trung bình bình phương sai số")
    print("  - ssim: Structural Similarity Index, đánh giá độ tương đồng về cấu trúc giữa ảnh dự đoán và ảnh thực")
    print("    Cú pháp: ssim hoặc ssim_<size> (ví dụ: ssim_21 để sử dụng mask size 21)")
    print("  - msssim: Multi-Scale SSIM, phiên bản đa tỷ lệ của SSIM")
    print("    Cú pháp: msssim hoặc msssim_<size>")
    print("  - bcl1: Binary Cross-Entropy + L1, kết hợp BCE và MAE")
    print("  - mae: Mean Absolute Error, tính trung bình sai số tuyệt đối")
    print("  - fss: Fractions Skill Score, đánh giá độ chính xác của dự báo theo không gian")
    print("    Cú pháp: fss hoặc fss_<mask_size>_<bins> (ví dụ: fss_5_0.1,0.5,0.9)")
    print("  - ks: Kolmogorov-Smirnov, đánh giá sự khác biệt giữa hai phân phối")
    print("    Cú pháp: ks hoặc ks_<mask_size> (ví dụ: ks_3)")
    print("  - coss: Cosine Similarity, đo độ tương đồng dựa trên góc giữa hai vector")
    print("\nVí dụ sử dụng:")
    print("  python cloudcast-unet.py --loss_function ssim")
    print("  python cloudcast-unet.py --loss_function fss_5_0.1,0.5,0.9")
    print("  python cloudcast-unet.py --loss_function ks_3")
    sys.exit(0)

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop_date", action="store", type=str)
    parser.add_argument("--cont", action="store_true")
    parser.add_argument("--n_channels", action="store", type=int, default=4)
    parser.add_argument("--loss_info", action="store_true", help="Hiển thị thông tin về các hàm loss và thoát")
    parser.add_argument(
        "--loss_function", action="store", type=str, default="ssim",
        help=(
            "Loss function: MeanSquaredError | ssim | msssim | bcl1 | fss | ks | coss | mae | "
            "ssim_mae[_wSSIM_wMAE] (e.g., ssim_mae or ssim_mae_0.5_0.5)"
        )
    )
    parser.add_argument(
        "--preprocess", action="store", type=str, default="img_size=512x512"
    )
    parser.add_argument("--label", action="store", type=str)
    # Tăng leadtime_conditioning từ 12 (15 phút) lên 18 (10 phút) để giữ cùng khoảng thời gian dự báo (3 giờ)
    parser.add_argument("--leadtime_conditioning", action="store", type=int, default=18)
    parser.add_argument("--reuse_y_as_x", action="store_true", default=False)
    parser.add_argument("--sequence_stride_minutes", action="store", type=int, default=10,
                      help="Khoảng thời gian giữa các chuỗi dữ liệu liên tiếp (phút)")
    parser.add_argument("--sequence_offset_minutes", action="store", type=int, default=0,
                      help="Độ lệch thời gian cho chuỗi dữ liệu (phút)")
    parser.add_argument("--learning_rate", action="store", type=float, default=None,
                      help="Tốc độ học cho quá trình huấn luyện. Nếu fine-tune mà không chỉ định, mặc định 1e-5")
    parser.add_argument("--checkpoint_path", action="store", type=str, default=None,
                      help="Đường dẫn weights để fine-tune (mặc định dùng 1e-5 nếu không chỉ định learning rate)")
    parser.add_argument("--no_mixed_precision", action="store_true", default=False,
                      help="Disable mixed precision training (default: enabled if GPUs available)")
    parser.add_argument("--force_load_weights", action="store_true", default=False,
                      help="Force load weights even if there are warnings (for fine-tuning)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--start_date", action="store", type=str)
    group.add_argument("--dataseries_file", action="store", type=str, default=None)
    group.add_argument("--dataseries_directory", action="store", type=str, default=None)

    args = parser.parse_args()
    
    if args.loss_info:
        print_loss_function_info()

    if args.label is not None:
        opts = CloudCastOptions(label=args.label)
    else:
        vars_ = vars(args)
        vars_["model"] = "unet"
        opts = CloudCastOptions(**vars_)

    if (
        (not args.start_date and not args.stop_date)
        and not args.dataseries_file
        and not args.dataseries_directory
    ):
        print(
            "Either start_date,stop_date or dataseries_file or dataseries_directory needs to be defined"
        )
        sys.exit(1)

    if args.start_date and args.stop_date:
        args.start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
        args.stop_date = datetime.datetime.strptime(args.stop_date, "%Y-%m-%d")

    return args, opts
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)



def get_batch_size(img_size):
    if img_size[0] >= 512:
        batch_size = 6
    elif img_size[0] >= 384:
        batch_size = 8
    elif img_size[0] >= 256:
        batch_size = 16
    elif img_size[0] >= 224:
        batch_size = 16
    elif img_size[0] >= 128:
        batch_size = 32
    else:
        batch_size = 64

    return batch_size


def with_dataset(m, args, opts):
    img_size = get_img_size(args.preprocess)

    lds = LazyDataSeries(
        img_size=img_size,
        batch_size=get_batch_size(img_size),
        training_mode=True,
        **vars(args),
    )

    # number of samples
    n = len(lds)
    # train-val split ratio
    r = 0.80
    # training dataset
    train_ds = lds.get_dataset(take_ratio=r)
    # validation dataset
    val_ds = lds.get_dataset(skip_ratio=r)
    # number of train data set steps (step = one batch)
    train_ds_steps = int((n * r) / lds.batch_size)
    # number of val data set steps
    val_ds_steps = int((n * (1 - r)) / lds.batch_size)

    print(
        "Total number of train dataset samples: {:d} number of steps: {:d} (batch_size: {:d})".format(
            int(n * r), train_ds_steps, lds.batch_size
        )
    )

    print(
        "Total number of validation samples: {:d} number of steps: {:d}".format(
            int(n * (1 - r)), val_ds_steps
        )
    )

    hist = m.fit(
        train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks(args, opts)
    )

    return hist


def callbacks(args, opts):
    # Đổi định dạng tệp lưu trữ thành .weights.h5 thay vì .ckpt để phù hợp với TensorFlow 2.19.1
    cp_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/{}/model.weights.h5".format(opts.get_label()),
        save_weights_only=True,
        save_best_only=True,
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=11, min_delta=0.0001, verbose=1
    )
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
    term_nan = keras.callbacks.TerminateOnNaN()

    return [cp_cb, early_stopping_cb, reduce_lr_cb, term_nan]


def save_model_info(args, opts, duration, hist, model_dir):
    now = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    with open(
        "{}/info-{}.json".format(model_dir, now),
        "w",
    ) as fp:
        data = {
            "args": vars(args),
            "opts": opts.__dict__,
            "duration": str(duration),
            "finished": now,
            "hostname": os.environ.get("HOSTNAME", "no-hostname"),
        }
        json.dump(data, fp, default=str)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.float32):
                return str(obj)
            return json.JSONEncoder.default(self, obj)

    with open(
        "{}/hist-{}.json".format(model_dir, now),
        "w",
    ) as fp:
        json.dump(hist, fp, cls=NumpyEncoder)


def run_model(args, opts):
    model_dir = "models/{}".format(opts.get_label())

    pretrained_weights = None
    if args.checkpoint_path:
        pretrained_weights = args.checkpoint_path
        print(f"🎯 FINE-TUNING FROM CHECKPOINT: {pretrained_weights}")
        if os.path.exists(pretrained_weights):
            print("✅ CHECKPOINT FILE EXISTS")
        else:
            print("❌ CHECKPOINT FILE NOT FOUND")
            print("🛑 STOPPING TRAINING - CHECKPOINT FILE REQUIRED FOR FINE-TUNING")
            sys.exit(1)
    elif args.cont:
        pretrained_weights = "checkpoints/{}/model.weights.h5".format(opts.get_label())
        print(f"🔄 CONTINUING FROM PREVIOUS WEIGHTS: {pretrained_weights}")
        if os.path.exists(pretrained_weights):
            print("✅ PREVIOUS WEIGHTS FILE EXISTS")
        else:
            print("❌ PREVIOUS WEIGHTS FILE NOT FOUND - WILL USE RANDOM WEIGHTS")

    img_size = get_img_size(opts.preprocess)
    n_channels = int(opts.n_channels)

    # Chỉ giữ lại leadtime_conditioning
    if opts.leadtime_conditioning:
        n_channels += 1

    # Hiển thị thông tin về hàm loss được sử dụng (sau khi có thể đã override)
    print(f"\nSử dụng hàm loss: {args.loss_function}")
    if args.loss_function.startswith("ssim_mae"):
        print("Combined Loss: SSIM + MAE with configurable weights (default 0.5/0.5)")
    elif args.loss_function.startswith("ssim"):
        print("Structural Similarity Index Loss - Đánh giá độ tương đồng về cấu trúc giữa ảnh")
    elif args.loss_function.startswith("msssim"):
        print("Multi-Scale SSIM Loss - Phiên bản đa tỷ lệ của SSIM")
    elif args.loss_function == "bcl1":
        print("Binary Cross-Entropy + L1 Loss - Kết hợp BCE và MAE")
    elif args.loss_function.startswith("fss"):
        print("Fractions Skill Score Loss - Đánh giá độ chính xác của dự báo theo không gian")
    elif args.loss_function.startswith("ks"):
        print("Kolmogorov-Smirnov Loss - Đánh giá sự khác biệt giữa hai phân phối")
    elif args.loss_function == "coss":
        print("Cosine Similarity Loss - Đo độ tương đồng dựa trên góc giữa hai vector")
    elif args.loss_function == "mae":
        print("Mean Absolute Error - Tính trung bình sai số tuyệt đối")
    elif args.loss_function == "MeanSquaredError":
        print("Mean Squared Error - Hàm loss cơ bản, tính trung bình bình phương sai số")
    print()
    
    # Configure GPUs: memory growth and distribution strategy
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    # Limit intra/inter op threads to avoid pthread_create failures on large models
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass
    if len(gpus) >= 2:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy on {strategy.num_replicas_in_sync} devices")
    else:
        strategy = tf.distribute.get_strategy()
        if len(gpus) == 1:
            print("Using single GPU")
        else:
            print("No GPU detected, using CPU")

    # Enable mixed precision based on arguments
    # Default: enable mixed precision if GPUs are available, unless explicitly disabled
    enable_mixed_precision = len(gpus) > 0 and not args.no_mixed_precision
    
    if enable_mixed_precision:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision enabled (mixed_float16)")
        except Exception as e:
            print(f"Could not enable mixed precision: {e}")
    else:
        print("Mixed precision disabled")

    # For fine-tuning, try to detect loss function from checkpoint path BEFORE building model
    if args.checkpoint_path and "ssim_mae" in args.checkpoint_path:
        detected_loss = "ssim_mae"
        print(f"Detected loss function from checkpoint path: {detected_loss}")
        # Override the loss function for fine-tuning
        args.loss_function = detected_loss
    
    # Determine learning rate: 1e-5 by default for fine-tuning, else 1e-3
    if args.learning_rate is not None:
        effective_lr = args.learning_rate
    else:
        effective_lr = 1e-5 if args.checkpoint_path else 1e-3

    # Build model under distribution strategy scope
    with strategy.scope():
        optimizer = keras.optimizers.Adam(learning_rate=effective_lr)
        print(f"Learning rate: {effective_lr}")
        try:
            m = unet(
                pretrained_weights,
                input_size=img_size + (n_channels,),
                loss_function=args.loss_function,
                optimizer=optimizer,
                force_load_weights=args.force_load_weights,
            )
        except (FileNotFoundError, RuntimeError) as e:
            print(f"🛑 TRAINING STOPPED: {e}")
            print("💡 SUGGESTIONS:")
            print("  1. Check if the checkpoint file path is correct")
            print("  2. Verify the checkpoint file exists and is accessible")
            print("  3. Use --force_load_weights if you want to continue with partial weights")
            sys.exit(1)

    start = datetime.datetime.now()

    hist = with_dataset(m, args, opts)

    duration = datetime.datetime.now() - start

    save_model(m, model_dir)
    save_model_info(args, opts, duration, hist.history, model_dir)

    # Save final checkpoint with timestamp YYYY_MM_DD_HHMM
    ts_env = os.environ.get("CLOUDCAST_TIMESTAMP")
    try:
        ts_final = ts_env if ts_env else datetime.now().strftime("%Y_%m_%d_%H%M")
    except Exception:
        ts_final = datetime.now().strftime("%Y_%m_%d_%H%M")
    final_ckpt_dir = f"checkpoints/{opts.get_label()}"
    os.makedirs(final_ckpt_dir, exist_ok=True)
    final_ckpt_path = f"{final_ckpt_dir}/model_{ts_final}.weights.h5"
    try:
        m.save_weights(final_ckpt_path)
        print(f"Saved final checkpoint: {final_ckpt_path}")
    except Exception as e:
        print(f"Failed to save final checkpoint to {final_ckpt_path}: {e}")
    
    # Lưu lịch sử huấn luyện
    with open(f"{model_dir}/history.json", "w") as f:
        json.dump(hist.history, f, cls=NumpyEncoder)

    print(f"Model training finished in {duration}")


if __name__ == "__main__":
    args, opts = parse_command_line()
    run_model(args, opts)
