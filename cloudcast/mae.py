import tensorflow as tf

def make_MAE_loss():
    @tf.function
    def my_mae_loss(y_true, y_pred):
        # Sử dụng hàm MeanAbsoluteError có sẵn của TensorFlow
        mae_lossfunction = tf.keras.losses.MeanAbsoluteError()
        return mae_lossfunction(y_true, y_pred)

    my_mae_loss.__name__ = "mean_absolute_error"
    return my_mae_loss
