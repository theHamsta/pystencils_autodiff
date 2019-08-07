import numpy as np
from pystencils_autodiff.backends import AVAILABLE_BACKENDS


def fix_layout(array, target_field, backend):
    assert array.shape == target_field.shape, "Array {}'s shape should be {} but is {}".format(
        target_field.name, target_field.shape, array.shape)
    assert backend.lower() in AVAILABLE_BACKENDS

    # Just index coordinate wrong?
    swapped_array = np.swapaxes(array, 0, -1)
    if (swapped_array.strides == target_field.strides and 
            swapped_array.shape == target_field.shade and 
            target_field.index_dimensions == 1):
        array = swapped_array

    # Everything ok
    everything_ok = (array.strides == target_field.strides
                     and array.shape == target_field.shape)

    if everything_ok:
        rtn = array
    else:  # no, fix it!
        f = target_field
        rtn = np.lib.stride_tricks.as_strided(np.zeros(f.shape, dtype=f.dtype.numpy_dtype),
                                              f.shape,
                                              [f.dtype.numpy_dtype.itemsize * a for a in f.strides])
        rtn[...] = array

    return rtn
