import tensorflow as tf

def make_MAE_loss():
    @tf.function
    def my_mae_loss(y_true, y_pred):
        # Use NONE reduction and manually compute mean for multi-GPU compatibility
        mae_lossfunction = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        return tf.reduce_mean(mae_lossfunction(y_true, y_pred))

    my_mae_loss.__name__ = "mean_absolute_error"
    return my_mae_loss
