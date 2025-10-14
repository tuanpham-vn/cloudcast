import tensorflow as tf


def make_bc_l1_loss():
    @tf.function
    def my_bc_l1_loss(y_true, y_pred):
        # Use per-example losses with Reduction.NONE, then average.
        bc_lossfunction = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
        mae_lossfunction = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )

        bc_loss = bc_lossfunction(y_true, y_pred)  # shape: (batch,)
        mae_loss = mae_lossfunction(y_true, y_pred)  # shape: (batch,)

        loss = tf.reduce_mean(bc_loss + mae_loss)
        return loss

    my_bc_l1_loss.__name__ = "binary_crossentropy-L1"
    return my_bc_l1_loss
