import tensorflow as tf

def make_SSIM_loss(mask_size=11, mask_sigma=1.5, k1=0.01, k2=0.03):
    @tf.function
    def SSIM_loss(y_true, y_pred):
        # Cast to float32 for numerical stability in SSIM computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        ssim_val = tf.image.ssim(
            y_true,
            y_pred,
            1.0,
            filter_size=mask_size,
            filter_sigma=mask_sigma,
            k1=k1,
            k2=k2,
        )
        loss = 1.0 - tf.reduce_mean(ssim_val)
        
        # Clip to prevent extreme values
        loss = tf.clip_by_value(loss, 0.0, 2.0)
        return loss

    SSIM_loss.__name__ = "SSIM"
    return SSIM_loss


def make_MS_SSIM_loss(mask_size=11, mask_sigma=1.5, k1=0.01, k2=0.03):
    @tf.function
    def MS_SSIM_loss(y_true, y_pred):
        # Cast to float32 for numerical stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        ms_ssim_val = tf.image.ssim_multiscale(
            y_true,
            y_pred,
            1.0,
            power_factors=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
            filter_size=mask_size,
            filter_sigma=mask_sigma,
            k1=k1,
            k2=k2,
        )
        loss = 1.0 - tf.reduce_mean(ms_ssim_val)
        
        # Clip to prevent extreme values
        loss = tf.clip_by_value(loss, 0.0, 2.0)
        return loss

    MS_SSIM_loss.__name__ = "MS_SSIM"
    return MS_SSIM_loss
