import numpy as np

from pystencils.utils import DotDict

try:
    import tensorflow as tf
except Exception:
    tf = None


def compute_gradient_error_without_border(x,
                                          x_shape,
                                          y,
                                          y_shape,
                                          num_border_pixels,
                                          ndim,
                                          x_init_value=None,
                                          delta=0.001,
                                          debug=False):
    """
    This function is a work-around for tf.compute_gradient_error that sets entries of Jacobi's matrix
    to zero that I believe belong to border regions of x or y.

    This may be necessary since `pystencils` leaves some ghost layer/boundary regions uninitialized.
    """

    jacobi_list = tf.compat.v1.test.compute_gradient(
        x, x_shape, y, y_shape, x_init_value, delta)

    if not isinstance(x_shape, list):
        x_shape = [x_shape, ]

    if isinstance(jacobi_list[0], np.ndarray):
        jacobi_list = [jacobi_list, ]

    max_error = 0
    avg_error = 0
    for i in range(len(jacobi_list)):
        diff = np.abs(jacobi_list[i][0] - jacobi_list[i][1])
        if debug:
            import pyconrad.autoinit
            pyconrad.imshow(jacobi_list[i][0], 'numerical')
            pyconrad.imshow(jacobi_list[i][1],
                            'automatic')
            pyconrad.imshow(
                jacobi_list[i][1] - jacobi_list[i][0], 'diff', wait_window_close=True)

        x_zero_matrix = np.ones(x_shape[i], np.bool)
        if num_border_pixels:
            if ndim == 2:
                x_zero_matrix[num_border_pixels:-num_border_pixels,
                              num_border_pixels:-num_border_pixels] = 0
            elif ndim == 3:
                x_zero_matrix[num_border_pixels:-num_border_pixels,
                              num_border_pixels:-num_border_pixels,
                              num_border_pixels:-num_border_pixels
                              ] = 0
            else:
                raise NotImplementedError()

        y_zero_matrix = np.ones(y_shape, np.bool)
        if num_border_pixels:
            if ndim == 2:
                y_zero_matrix[num_border_pixels:-num_border_pixels,
                              num_border_pixels:-num_border_pixels] = 0
            elif ndim == 3:
                y_zero_matrix[num_border_pixels:-num_border_pixels,
                              num_border_pixels:-num_border_pixels,
                              num_border_pixels:-num_border_pixels
                              ] = 0
            else:
                raise NotImplementedError()

            zeros = x_zero_matrix.flatten(
            ).reshape(-1, 1) @ y_zero_matrix.flatten()[None]

            diff[zeros] = 0
        max_error = max(max_error, np.max(diff))
        avg_error += np.mean(diff)

    avg_error /= len(jacobi_list)

    return DotDict({'max_error': max_error, 'avg_error': avg_error})
