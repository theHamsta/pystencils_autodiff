try:
    import tensorflow as tf
except Exception:
    pass


def masked_mse(a, b, mask):
    """ Mean squared error within mask """
    return tf.losses.mean_pairwise_squared_error(tf.boolean_mask(a, mask), tf.boolean_mask(b, mask))


def total_var(tensor, norm=2):
    pixel_dif1 = tensor[1:, :] - tensor[:-1, :]
    pixel_dif2 = tensor[:, 1:] - tensor[:, :-1]
    if norm == 2:
        return tf.reduce_sum(pixel_dif1 * pixel_dif1) + tf.reduce_sum(pixel_dif2 * pixel_dif2)
    if norm == 1:
        return tf.reduce_sum(tf.abs(pixel_dif1)) + tf.reduce_sum(tf.abs(pixel_dif2))


def mean_total_var(tensor, norm=2):
    pixel_dif1 = tensor[1:, :] - tensor[:-1, :]
    pixel_dif2 = tensor[:, 1:] - tensor[:, :-1]
    if norm == 2:
        return (tf.reduce_mean(pixel_dif1 * pixel_dif1) + tf.reduce_mean(pixel_dif2 * pixel_dif2)) / 2
    if norm == 1:
        return (tf.reduce_mean(tf.abs(pixel_dif1)) + tf.reduce_mean(tf.abs(pixel_dif2))) / 2
