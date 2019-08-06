import numpy as np

try:
    import tensorflow as tf
except ImportError:
    pass
try:
    import torch
except ImportError:
    pass


def tf_constant_from_field(field, init_val=0):
    return tf.constant(init_val, dtype=field.dtype.numpy_dtype, shape=field.shape, name=field.name + '_constant')


def tf_scalar_variable_from_field(field, init_val, constraint=None):
    var = tf.Variable(init_val, dtype=field.dtype.numpy_dtype, name=field.name + '_variable', constraint=constraint)
    return var * tf_constant_from_field(field, 1)


def tf_variable_from_field(field, init_val=0, constraint=None):
    if isinstance(init_val, (int, float)):
        init_val *= np.ones(field.shape, field.dtype.numpy_dtype)

    return tf.Variable(init_val, dtype=field.dtype.numpy_dtype, name=field.name + '_variable', constraint=constraint)


def tf_placeholder_from_field(field):
    return tf.placeholder(dtype=field.dtype.numpy_dtype, name=field.name + '_placeholder', shape=field.shape)


def torch_tensor_from_field(field, init_val=0, cuda=True, requires_grad=False):
    if isinstance(init_val, (int, float)):
        init_val *= np.ones(field.shape, field.dtype.numpy_dtype)
    device = torch.device('cuda' if cuda else 'cpu')
    return torch.tensor(init_val, requires_grad=requires_grad, device=device)
