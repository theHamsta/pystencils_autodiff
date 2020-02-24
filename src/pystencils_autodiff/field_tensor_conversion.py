import numpy as np
import sympy

from pystencils import Field
from pystencils.field import FieldType


class _WhatEverClass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ArrayWrapper:
    def __init__(self,
                 array,
                 index_dimensions=0,
                 field_type=FieldType.GENERIC,
                 coordinate_transform=None,
                 spacing=None,
                 origin=None,
                 coordinate_origin=None):
        self.array = array
        self.index_dimensions = index_dimensions
        self.field_type = field_type
        if spacing:
            coordinate_transform = sympy.diag(*spacing)
        if origin:
            origin = sympy.Matrix(origin)
            spacing = spacing or origin / origin
            coordinate_transform = sympy.diag(*spacing)
            coordinate_origin = origin / spacing

        if coordinate_transform:
            self.coordinate_transform = coordinate_transform
        if coordinate_origin:
            self.coordinate_origin = coordinate_origin

    def __array__(self):
        return self.array

    def __getattr__(self, name):
        return getattr(self.array, name)


def _torch_tensor_to_numpy_shim(tensor):

    from pystencils.autodiff.backends._pytorch import torch_dtype_to_numpy
    fake_array = _WhatEverClass(
        strides=[tensor.stride(i) * tensor.storage().element_size() for i in range(len(tensor.shape))],
        shape=tensor.shape,
        dtype=torch_dtype_to_numpy(tensor.dtype))
    return fake_array


def create_field_from_array_like(field_name, maybe_array, annotations=None):
    if annotations and isinstance(annotations, dict):
        index_dimensions = annotations.get('index_dimensions', 0)
        field_type = annotations.get('field_type', FieldType.GENERIC)
    elif isinstance(maybe_array, ArrayWrapper):
        index_dimensions = maybe_array.index_dimensions
        field_type = maybe_array.field_type
        maybe_array = maybe_array.array
    else:
        index_dimensions = 0
        field_type = FieldType.GENERIC

    if 'tensorflow' in str(type(maybe_array)) and 'Tensor' in str(type(maybe_array)):
        try:
            # This fails on eager execution
            return Field.create_fixed_size(maybe_array.name or field_name,
                                           maybe_array.shape,
                                           index_dimensions=index_dimensions,
                                           dtype=maybe_array.dtype.as_numpy_dtype())
        except Exception:
            return Field.create_fixed_size(field_name,
                                           maybe_array.shape,
                                           index_dimensions=index_dimensions,
                                           dtype=maybe_array.dtype.as_numpy_dtype())
    elif 'torch.Tensor' in str(type(maybe_array)):
        maybe_array = _torch_tensor_to_numpy_shim(maybe_array)

    field = Field.create_from_numpy_array(field_name, maybe_array, index_dimensions)
    field.field_type = field_type
    if hasattr(maybe_array, 'coordinate_transform'):
        field.coordinate_transform = maybe_array.coordinate_transform
    if hasattr(maybe_array, 'coordinate_origin'):
        field.coordinate_origin = maybe_array.coordinate_origin
    return field


def coerce_to_field(field_name, array_like):
    if isinstance(array_like, Field):
        return array_like.new_field_with_different_name(field_name, array_like)
    return create_field_from_array_like(field_name, array_like)


def is_array_like(a):
    import pycuda.gpuarray
    return (hasattr(a, '__array__')
            or isinstance(a, pycuda.gpuarray.GPUArray)
            or ('tensorflow' in str(type(a)) and 'Tensor' in str(type(a)))
            or 'torch.Tensor' in str(type(a))) and not isinstance(a,
                                                                  (sympy.Matrix,
                                                                   sympy.MutableDenseMatrix,
                                                                   sympy.MatrixSymbol))


def tf_constant_from_field(field, init_val=0):
    import tensorflow as tf
    return tf.constant(init_val, dtype=field.dtype.numpy_dtype, shape=field.shape, name=field.name + '_constant')


def tf_scalar_variable_from_field(field, init_val, constraint=None):
    import tensorflow as tf
    var = tf.Variable(init_val, dtype=field.dtype.numpy_dtype, name=field.name + '_variable', constraint=constraint)
    return var * tf_constant_from_field(field, 1)


def tf_variable_from_field(field, init_val=0, constraint=None):
    import tensorflow as tf
    if isinstance(init_val, (int, float)):
        init_val *= np.ones(field.shape, field.dtype.numpy_dtype)

    return tf.Variable(init_val, dtype=field.dtype.numpy_dtype, name=field.name + '_variable', constraint=constraint)


def tf_placeholder_from_field(field):
    import tensorflow as tf
    return tf.placeholder(dtype=field.dtype.numpy_dtype, name=field.name + '_placeholder', shape=field.shape)


def torch_tensor_from_field(field, init_val=0, cuda=True, requires_grad=False):
    import torch
    if isinstance(init_val, (int, float)):
        init_val *= np.ones(field.shape, field.dtype.numpy_dtype)
    device = torch.device('cuda' if cuda else 'cpu')
    return torch.tensor(init_val, requires_grad=requires_grad, device=device)
