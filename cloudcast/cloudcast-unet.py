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



def get_total_batch_size(img_size):
    if img_size[0] >= 512:
        total_batch_size = 128
    elif img_size[0] >= 384:
        total_batch_size = 128
    elif img_size[0] >= 256:
        total_batch_size = 128
    elif img_size[0] >= 224:
        total_batch_size = 128
    elif img_size[0] >= 128:
        total_batch_size = 128
    else:
        total_batch_size = 128

    return total_batch_size


def with_dataset(m, args, opts, per_replica_batch_size=None):
    img_size = get_img_size(args.preprocess)
    
    if per_replica_batch_size is not None:
        batch_size = per_replica_batch_size
        print(f"✓ Using per-replica batch size from run_model: {batch_size}")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) >= 2:
            total_batch_size = get_total_batch_size(img_size)
            num_gpus = len(gpus)
            if total_batch_size % num_gpus != 0:
                adjusted_total_batch_size = (total_batch_size // num_gpus) * num_gpus
                if adjusted_total_batch_size == 0:
                    adjusted_total_batch_size = num_gpus
                print(f"⚠️  Adjusting total batch size from {total_batch_size} to {adjusted_total_batch_size} for {num_gpus} GPUs")
                batch_size = adjusted_total_batch_size // num_gpus
            else:
                batch_size = total_batch_size // num_gpus
        else:
            total_batch_size = get_total_batch_size(img_size)
            batch_size = total_batch_size
            print(f"✓ Single GPU batch size: {batch_size}")

    lds = LazyDataSeries(
        img_size=img_size,
        batch_size=batch_size,
        training_mode=True,
        **vars(args),
    )

    n = len(lds)
    r = 0.80
    train_ds = lds.get_dataset(take_ratio=r)
    val_ds = lds.get_dataset(skip_ratio=r)
    
    train_samples = int(n * r)
    val_samples = int(n * (1 - r))
    
    if train_samples < lds.batch_size:
        print(f"Warning: Not enough training samples ({train_samples}) for batch size {lds.batch_size}")
        print("Adjusting batch size for training...")
        lds.batch_size = max(1, train_samples)
    
    if val_samples < lds.batch_size:
        print(f"Warning: Not enough validation samples ({val_samples}) for batch size {lds.batch_size}")
        print("Adjusting batch size for validation...")
        val_batch_size = max(1, val_samples)
    else:
        val_batch_size = lds.batch_size
    
    if train_samples == 0:
        raise ValueError("No training samples available! Check your dataset.")
    if val_samples == 0:
        print("Warning: No validation samples available. Using training data for validation.")
        val_samples = train_samples
        val_batch_size = lds.batch_size
    
    train_ds = lds.get_dataset(take_ratio=r)
    val_ds = lds.get_dataset(skip_ratio=r)
    
    train_ds_steps = max(1, train_samples // lds.batch_size)
    val_ds_steps = max(1, val_samples // val_batch_size)

    print(
        "Total number of train dataset samples: {:d} number of steps: {:d} (batch_size: {:d})".format(
            train_samples, train_ds_steps, lds.batch_size
        )
    )

    print(
        "Total number of validation samples: {:d} number of steps: {:d} (batch_size: {:d})".format(
            val_samples, val_ds_steps, val_batch_size
        )
    )
    
    print("Verifying dataset pipeline...")
    try:
        sample_count = 0
        for x, y in train_ds.take(3):
            sample_count += 1
            print(f"  Sample {sample_count}: x.shape={x.shape}, y.shape={y.shape}, "
                  f"x range=[{tf.reduce_min(x):.4f}, {tf.reduce_max(x):.4f}], "
                  f"y range=[{tf.reduce_min(y):.4f}, {tf.reduce_max(y):.4f}]")
        print("✓ Dataset pipeline verified")
        
        print("Checking batch consistency...")
        for i, (x, y) in enumerate(train_ds.take(5)):
            if x.shape[0] != lds.batch_size:
                print(f"⚠️  Warning: Batch {i} has {x.shape[0]} samples instead of {lds.batch_size}")
            if i >= 4:
                break
        print("✓ Batch consistency verified")
        
    except Exception as e:
        print(f"✗ Dataset pipeline verification failed: {e}")
        print("This might be due to insufficient samples or batch size mismatch.")
        print("Try reducing batch size or increasing dataset size.")
        raise

    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) >= 2:
        print(f"✓ Using {len(gpus)} GPUs - MirroredStrategy will automatically distribute steps")
    else:
        print(f"✓ Using {len(gpus)} GPU(s) - standard training")

    hist = m.fit(
        train_ds, 
        steps_per_epoch=train_ds_steps,
        epochs=EPOCHS, 
        validation_data=val_ds,
        validation_steps=val_ds_steps,
        callbacks=callbacks(args, opts)
    )

    return hist


class NaNLossCallback(keras.callbacks.Callback):
    """Callback to handle NaN/Inf losses and zero losses"""
    def __init__(self):
        super().__init__()
        self.nan_epochs = []
        self.zero_loss_count = 0
    
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print(f"\n⚠️  Detected NaN/Inf loss at batch {batch}. Stopping epoch early.")
                self.model.stop_training = True
            elif loss == 0.0 and batch > 5:
                # Zero loss after first few batches is suspicious
                print(f"\n⚠️  Warning: Zero loss detected at batch {batch}. This may indicate data pipeline issue.")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.zero_loss_count = 0
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        
        if loss is not None and loss == 0.0:
            self.zero_loss_count += 1
            print(f"⚠️  Epoch {epoch + 1} had ZERO loss. This indicates data pipeline issue!")
            if self.zero_loss_count >= 3:
                print(f"🛑  Multiple consecutive zero loss epochs detected. Stopping training.")
                self.model.stop_training = True
        else:
            self.zero_loss_count = 0
        
        if (loss is not None and (np.isnan(loss) or np.isinf(loss))) or \
           (val_loss is not None and (np.isnan(val_loss) or np.isinf(val_loss))):
            self.nan_epochs.append(epoch)
            print(f"⚠️  Epoch {epoch + 1} had NaN/Inf loss. Will continue with next epoch.")
            self.model.stop_training = False

def callbacks(args, opts):
    cp_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/{}/model.weights.h5".format(opts.get_label()),
        save_weights_only=True,
        save_best_only=True,
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=13, min_delta=0.0001, verbose=1
    )
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5)
    nan_callback = NaNLossCallback()

    return [cp_cb, early_stopping_cb, reduce_lr_cb, nan_callback]


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
    
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    
    enable_mixed_precision = len(gpus) > 0 and not args.no_mixed_precision
    if enable_mixed_precision:
        try:
            from tensorflow.keras import mixed_precision
            if args.loss_function.startswith("ssim") or args.loss_function.startswith("msssim"):
                try:
                    mixed_precision.set_global_policy('mixed_bfloat16')
                    print("✓ Mixed precision enabled (mixed_bfloat16 for SSIM stability)")
                except Exception:
                    print("✓ Mixed precision disabled for SSIM loss (using float32 for stability)")
                    enable_mixed_precision = False
            else:
                mixed_precision.set_global_policy('mixed_float16')
                print("✓ Mixed precision enabled (mixed_float16)")
        except Exception as e:
            print(f"✗ Could not enable mixed precision: {e}")
            enable_mixed_precision = False
    
    if not enable_mixed_precision:
        print("✓ Mixed precision disabled (float32)")
    
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass
    
    if len(gpus) >= 2:
        strategy = tf.distribute.MirroredStrategy()
        print(f"✓ Using MirroredStrategy on {strategy.num_replicas_in_sync} GPUs")
        total_batch_size = get_total_batch_size(get_img_size(opts.preprocess))
        per_replica_batch_size = total_batch_size // strategy.num_replicas_in_sync
        global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
        print(f"✓ Per-replica batch size: {per_replica_batch_size} (each GPU processes this many samples)")
        print(f"✓ Global batch size: {global_batch_size} (total samples across all GPUs)")
        print(f"✓ Effective batch size per step: {global_batch_size} (all GPUs combined)")
        
        if per_replica_batch_size % strategy.num_replicas_in_sync != 0:
            adjusted_batch_size = (per_replica_batch_size // strategy.num_replicas_in_sync) * strategy.num_replicas_in_sync
            if adjusted_batch_size == 0:
                adjusted_batch_size = strategy.num_replicas_in_sync
            print(f"⚠️  Adjusting batch size from {per_replica_batch_size} to {adjusted_batch_size} for multi-GPU compatibility")
            per_replica_batch_size = adjusted_batch_size
    else:
        strategy = tf.distribute.get_strategy()
        total_batch_size = get_total_batch_size(get_img_size(opts.preprocess))
        per_replica_batch_size = total_batch_size
        if len(gpus) == 1:
            print("✓ Using single GPU")
            print(f"✓ Batch size: {per_replica_batch_size}")
        else:
            print("✓ No GPU detected, using CPU")
            print(f"✓ Batch size: {per_replica_batch_size}")
    
    policy = tf.keras.mixed_precision.global_policy()
    print(f"✓ Compute dtype: {policy.compute_dtype}, Variable dtype: {policy.variable_dtype}")

    if args.checkpoint_path and "ssim_mae" in args.checkpoint_path:
        detected_loss = "ssim_mae"
        print(f"Detected loss function from checkpoint path: {detected_loss}")
        args.loss_function = detected_loss
    
    if args.learning_rate is not None:
        effective_lr = args.learning_rate
    else:
        if args.checkpoint_path:
            effective_lr = 1e-5
        elif args.loss_function.startswith("ssim") or args.loss_function.startswith("msssim"):
            effective_lr = 5e-4
        else:
            effective_lr = 1e-3

    with strategy.scope():
        optimizer = keras.optimizers.Adam(
            learning_rate=effective_lr,
            clipnorm=1.0
        )
        print(f"Learning rate: {effective_lr}")
        print(f"Gradient clipping: clipnorm=1.0")
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

    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) >= 2:
        total_batch_size = get_total_batch_size(get_img_size(opts.preprocess))
        per_replica_batch_size = total_batch_size // len(gpus)
        if total_batch_size % len(gpus) != 0:
            adjusted_batch_size = (total_batch_size // len(gpus)) * len(gpus)
            if adjusted_batch_size == 0:
                adjusted_batch_size = len(gpus)
            per_replica_batch_size = adjusted_batch_size // len(gpus)
    else:
        total_batch_size = get_total_batch_size(get_img_size(opts.preprocess))
        per_replica_batch_size = total_batch_size
    
    hist = with_dataset(m, args, opts, per_replica_batch_size)

    duration = datetime.datetime.now() - start

    model_save_path = f"{model_dir}/model.keras"
    os.makedirs(model_dir, exist_ok=True)
    try:
        m.save(model_save_path)
        print(f"Saved model to: {model_save_path}")
    except Exception as e:
        print(f"Warning: Could not save full model: {e}")
        print("Saving weights only instead...")
        m.save_weights(f"{model_dir}/model.weights.h5")
    
    save_model_info(args, opts, duration, hist.history, model_dir)

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
    
    with open(f"{model_dir}/history.json", "w") as f:
        json.dump(hist.history, f, cls=NumpyEncoder)

    print(f"Model training finished in {duration}")


if __name__ == "__main__":
    args, opts = parse_command_line()
    run_model(args, opts)
