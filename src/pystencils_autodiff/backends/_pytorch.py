import uuid

import numpy as np
try:
    import torch
except ImportError:
    pass

try:
    import pycuda.autoinit
    import pycuda.gpuarray
    import pycuda.driver
    HAS_PYCUDA = True
except Exception:
    HAS_PYCUDA = False


def create_autograd_function(autodiff_obj, inputfield_to_tensor_dict, forward_loop, backward_loop,
                             convert_tensors_to_arrays=True):
    field_to_tensor_dict = inputfield_to_tensor_dict
    backward_input_fields = autodiff_obj.backward_input_fields

    # Allocate output tensor for forward and backward pass
    for field in autodiff_obj.forward_output_fields + autodiff_obj.backward_output_fields:
        field_to_tensor_dict[field] = torch.zeros(
            *field.shape,
            dtype=numpy_dtype_to_torch(field.dtype.numpy_dtype),
            device=list(inputfield_to_tensor_dict.values())[0].device)

    tensor_to_field_dict = {
        v: k for k, v in field_to_tensor_dict.items()}

    def _tensors_to_dict(is_cuda, args, additional_dict={}):
        arrays = dict()
        lookup_dict = {**tensor_to_field_dict, **additional_dict}
        for a in args:

            if convert_tensors_to_arrays:
                if is_cuda:
                    a.cuda()
                    array = tensor_to_gpuarray(a)
                else:
                    a.cpu()
                    array = a.data.numpy()

                try:
                    arrays[lookup_dict[a].name] = array
                except Exception as e:
                    raise e

            else:
                array = a
                try:
                    arrays[lookup_dict[a].name] = array
                except Exception as e:
                    raise e

        return arrays

    def forward(self, *input_tensors):

        self.save_for_backward(*input_tensors)
        all_tensors = field_to_tensor_dict.values()

        is_cuda = all(a.is_cuda for a in all_tensors)
        arrays = _tensors_to_dict(is_cuda, all_tensors)

        forward_loop(**arrays, is_cuda=is_cuda)

        return tuple(field_to_tensor_dict[f] for f in autodiff_obj.forward_output_fields)

    def backward(self, *grad_outputs):
        all_tensors = grad_outputs + tuple(field_to_tensor_dict.values())

        is_cuda = all(a.is_cuda for a in all_tensors)
        arrays = _tensors_to_dict(is_cuda, all_tensors, additional_dict={
            grad_outputs[i]: f for i, f in enumerate(backward_input_fields)})
        backward_loop(**arrays, is_cuda=is_cuda)
        return tuple(field_to_tensor_dict[f] for f in autodiff_obj.backward_output_fields)

    cls = type(str(uuid.uuid4()), (torch.autograd.Function,), {})
    cls.forward = forward
    cls.backward = backward
    return cls


# from: https://stackoverflow.com/questions/51438232/how-can-i-create-a-pycuda-gpuarray-from-a-gpu-memory-address


def torch_dtype_to_numpy(dtype):
    dtype_name = str(dtype).replace('torch.', '')     # remove 'torch.'
    return getattr(np, dtype_name)


def numpy_dtype_to_torch(dtype):
    dtype_name = str(dtype)
    return getattr(torch, dtype_name)


# Fails if different context/thread
def tensor_to_gpuarray(tensor):
    if not tensor.is_cuda:
        raise ValueError(
            'Cannot convert CPU tensor to GPUArray (call `cuda()` on it)')
    else:
        return pycuda.gpuarray.GPUArray(tensor.shape,
                                        dtype=torch_dtype_to_numpy(tensor.dtype),
                                        gpudata=tensor.data_ptr())


def gpuarray_to_tensor(gpuarray, context=None):
    '''Convert a :class:`pycuda.gpuarray.GPUArray` to a :class:`torch.Tensor`. The underlying
    storage will NOT be shared, since a new copy must be allocated.
    Parameters
    ----------
    gpuarray  :   pycuda.gpuarray.GPUArray
    Returns
    -------
    torch.Tensor
    '''
    if not context:
        context = pycuda.autoinit.context
    shape = gpuarray.shape
    dtype = gpuarray.dtype
    out_dtype = dtype
    out = torch.zeros(shape, dtype=out_dtype).cuda()
    gpuarray_copy = tensor_to_gpuarray(out)
    byte_size = gpuarray.itemsize * gpuarray.size
    pycuda.driver.memcpy_dtod(gpuarray_copy.gpudata,
                              gpuarray.gpudata, byte_size)
    return out


if HAS_PYCUDA:
    class GpuPointerHolder(pycuda.driver.PointerHolderBase):

        def __init__(self, tensor):
            super().__init__()
            self.tensor = tensor
            self.gpudata = tensor.data_ptr()

        def get_pointer(self):
            return self.tensor.data_ptr()

        def __int__(self):
            return self.__index__()

        # without an __index__ method, arithmetic calls to the GPUArray backed by this pointer fail
        # not sure why, this needs to return some integer, apparently
        def __index__(self):
            return self.gpudata
else:
    GpuPointerHolder = None
