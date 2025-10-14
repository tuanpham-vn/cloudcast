import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    MaxPooling2D,
    UpSampling2D,
    Cropping2D,
    Concatenate,
    ConvLSTM2D,
    BatchNormalization,
    Conv3D,
    Activation,
)
from tensorflow.keras.models import Model

from fss import make_FSS_loss
from ssim import make_SSIM_loss, make_MS_SSIM_loss
from ks import make_KS_loss
from bcl1 import make_bc_l1_loss
from mae import make_MAE_loss

from tensorflow.keras import mixed_precision
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x for x in local_device_protos if x.device_type == "GPU"]


def get_compute_capability(gpu_id=0):
    devices = tf.config.experimental.list_physical_devices()

    for d in devices:
        if d[1] == "GPU" and int(d[0][-1]) == gpu_id:
            details = tf.config.experimental.get_device_details(d)
            return details["compute_capability"]

    return None


cc = get_compute_capability()

if cc is not None and int(cc[0]) >= 7:
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)

policy = tf.keras.mixed_precision.global_policy()

print(
    "Compute dtype: {} Variable dtype: {} Number of GPUs: {}".format(
        policy.compute_dtype, policy.variable_dtype, len(get_available_gpus())
    )
)


def get_loss_function(loss_function):
    # Combined losses
    if loss_function.startswith("ssim_mae"):
        # Syntax options:
        #  - "ssim_mae" -> defaults to weights 0.5, 0.5
        #  - "ssim_mae_w1_w2" -> e.g. ssim_mae_0.7_0.3
        parts = loss_function.split("_")
        if len(parts) == 3:
            try:
                w_ssim = float(parts[1])
                w_mae = float(parts[2])
            except Exception:
                w_ssim, w_mae = 0.5, 0.5
        else:
            w_ssim, w_mae = 0.5, 0.5

        ssim_loss = make_SSIM_loss()
        mae_loss = make_MAE_loss()

        def combined_loss(y_true, y_pred):
            return w_ssim * ssim_loss(y_true, y_pred) + w_mae * mae_loss(y_true, y_pred)

        combined_loss.__name__ = f"SSIM_MAE_combined_{w_ssim}_{w_mae}"
        return combined_loss

    if loss_function.startswith("ssim"):
        values = loss_function.split("_")
        if len(values) == 1:
            return make_SSIM_loss()

        return make_SSIM_loss(int(values[1]))
    elif loss_function.startswith("msssim"):
        values = loss_function.split("_")
        if len(values) == 1:
            return make_MS_SSIM_loss()

        return make_SSIM_loss(int(values[1]))
    elif loss_function == "bcl1":
        return make_bc_l1_loss()
    elif loss_function == "mae":
        return make_MAE_loss()
    elif loss_function.startswith("fss"):
        values = loss_function.split("_")
        if len(values) == 1:
            return make_FSS_loss(5)
        assert len(values) == 3

        mask = int(values[1])
        bins = list(map(lambda x: float(x), values[2].split(",")))

        b = []
        for i in range(len(bins) - 1):
            b.append([bins[i], bins[i + 1]])

        bins = tf.constant(b)
        return make_FSS_loss(mask, bins, hard_discretization=False)
    elif loss_function.startswith("ks"):
        values = loss_function.split("_")
        if len(values) == 1:
            return make_KS_loss()

        return make_KS_loss(int(values[1]))
    elif loss_function == "coss":
        ngpu = len(get_available_gpus())

        def coss(yt, yp):
            lf = tf.keras.losses.CosineSimilarity(
                reduction=tf.keras.losses.Reduction.NONE
            )
            loss = lf(tf.expand_dims(yt, -1), tf.expand_dims(yp, -1))
            loss = tf.reduce_mean(loss) * (1.0 / ngpu)
            return loss

        return coss

    return loss_function


def get_metrics():
    return [
        "RootMeanSquaredError",
        "MeanAbsoluteError",
    ]  # , make_FSS_loss(20, 0), make_SSIM_loss(21), make_KS_loss(21)]

def get_combined_metrics():
    """Get metrics for combined loss functions"""
    ssim_metric = make_SSIM_loss()
    mae_metric = make_MAE_loss()
    return [ssim_metric, mae_metric]

def get_all_metrics():
    """Get all available metrics for comparison during fine-tuning"""
    ssim_metric = make_SSIM_loss()
    mae_metric = make_MAE_loss()
    bcl1_metric = make_bc_l1_loss()
    return [ssim_metric, mae_metric, bcl1_metric]


def unet(
    pretrained_weights=None,
    input_size=(256, 256, 1),
    loss_function="MeanSquaredError",
    optimizer="adam",
    n_categories=None,
    compile=True
):
    inputs = Input(input_size)

    def conv_block(inp, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(inp)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def encoder_block(inp, num_filters):
        x = conv_block(inp, num_filters)
        p = MaxPooling2D((2, 2))(x)

        return x, p

    def decoder_block(inp, skip_connections, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inp)
        x = Concatenate()([x, skip_connections])
        x = conv_block(x, num_filters)

        return x

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Force datatype of output layer to float32; float16 is not numerically
    # stable enough in this layer

    if n_categories is None:
        outputs = Conv2D(1, 1, padding="same", activation="sigmoid", dtype="float32")(
            d4
        )
    else:
        outputs = Conv2D(
            n_categories, 1, padding="same", activation="softmax", dtype="float32"
        )(d4)

    assert n_categories is None or loss_function == "sparse_categorical_crossentropy"

    model = Model(inputs, outputs)

    if compile:
        # Choose metrics: if SSIM-based loss is used, also report SSIM and MAE metrics
        metrics = get_metrics()
        try:
            lf_name = loss_function if isinstance(loss_function, str) else ""
        except Exception:
            lf_name = ""
        if isinstance(lf_name, str) and (
            lf_name.startswith("ssim")
            or lf_name.startswith("msssim")
            or lf_name.startswith("ssim_mae")
        ):
            metrics = get_combined_metrics()

        model.compile(
            optimizer=optimizer,
            loss=get_loss_function(loss_function),
            metrics=metrics,
        )

    if pretrained_weights is not None:
        try:
            # Try loading weights only (skip optimizer state)
            model.load_weights(pretrained_weights, by_name=True, skip_mismatch=True)
            print(f"Successfully loaded weights from {pretrained_weights}")
        except Exception as e:
            print(f"Warning: Could not load weights from {pretrained_weights}: {e}")
            print("Continuing with random weights...")

    return model
