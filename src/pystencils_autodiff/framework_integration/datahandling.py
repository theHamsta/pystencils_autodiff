# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

try:
    import torch
except ImportError:
    torch = None
from typing import Sequence, Tuple, Union

import numpy as np

import pystencils
from pystencils.autodiff.backends._pytorch import numpy_dtype_to_torch
from pystencils.field import (
    Field, FieldType, create_numpy_array_with_layout, layout_string_to_tuple,
    spatial_layout_string_to_tuple)


class MultiShapeDatahandling(pystencils.datahandling.SerialDataHandling):
    """
    Specialization of :class:`pystencils.datahandling.SerialDataHandling` to support arrays with different sizes.
    """

    def __init__(self,
                 default_domain_size: Sequence[int],
                 default_ghost_layers: int = 0,
                 default_layout: str = 'numpy',
                 periodicity: Union[bool, Sequence[bool]] = False,
                 default_target: str = 'cpu',
                 opencl_queue=None,
                 opencl_ctx=None,
                 array_handler=None) -> None:
        """
        Same as :func:`pystencils.datahandling.SerialDataHandling.__init__` with better defaults for communication
        free applications
        """
        super().__init__(
            default_domain_size,
            default_ghost_layers,
            default_layout,
            periodicity,
            default_target,
            opencl_queue,
            opencl_ctx,
            array_handler=None)

    def add_arrays(self, description: str, spatial_shape=None) -> Tuple[pystencils.Field]:
        from pystencils.field import _parse_part1, _parse_description

        if ':' in description:
            fields_info, data_type, size = _parse_description(description)
            names = []
            for name, indices in fields_info:
                names.append(name)
                self.add_array(name, values_per_cell=indices, dtype=data_type, spatial_shape=size)

            return (self.fields[n] for n in names)
        else:
            names = []
            for name, indices in _parse_part1(description):
                names.append(name)
                self.add_array(name, values_per_cell=indices, dtype=np.float32, spatial_shape=spatial_shape)

            return (self.fields[n] for n in names)

    def add_array(self, name, values_per_cell=1, dtype=np.float32, latex_name=None, ghost_layers=None, layout=None,
                  cpu=True, gpu=None, alignment=False, field_type=FieldType.GENERIC, spatial_shape=None):
        if ghost_layers is None:
            ghost_layers = self.default_ghost_layers
        if layout is None:
            layout = self.default_layout
        if gpu is None:
            gpu = self.default_target in self._GPU_LIKE_TARGETS

        kwargs = {
            'shape': tuple(s + 2 * ghost_layers for s in (spatial_shape if spatial_shape else self._domainSize)),
            'dtype': dtype,
        }

        if not hasattr(values_per_cell, '__len__'):
            values_per_cell = (values_per_cell, )
        if len(values_per_cell) == 1 and values_per_cell[0] == 1:
            values_per_cell = ()

        self._field_information[name] = {
            'ghost_layers': ghost_layers,
            'values_per_cell': values_per_cell,
            'layout': layout,
            'dtype': dtype,
            'alignment': alignment,
        }

        index_dimensions = len(values_per_cell)
        kwargs['shape'] = kwargs['shape'] + values_per_cell

        if index_dimensions > 0:
            layout_tuple = layout_string_to_tuple(layout, self.dim + index_dimensions)
        else:
            layout_tuple = spatial_layout_string_to_tuple(layout, self.dim)

        # cpu_arr is always created - since there is no create_pycuda_array_with_layout()
        byte_offset = ghost_layers * np.dtype(dtype).itemsize
        numpy_array = create_numpy_array_with_layout(layout=layout_tuple, alignment=alignment,
                                                     byte_offset=byte_offset, **kwargs)
        cpu_arr = self.array_handler.from_numpy(numpy_array)

        if alignment and gpu:
            raise NotImplementedError("Alignment for GPU fields not supported")

        if cpu:
            if name in self.cpu_arrays:
                raise ValueError("CPU Field with this name already exists")
            self.cpu_arrays[name] = cpu_arr
        if gpu:
            if name in self.gpu_arrays:
                raise ValueError("GPU Field with this name already exists")
            self.gpu_arrays[name] = self.array_handler.to_gpu(cpu_arr)

        assert all(f.name != name for f in self.fields.values()), "Symbolic field with this name already exists"
        self.fields[name] = Field.create_from_numpy_array(name,
                                                          numpy_array,
                                                          index_dimensions=index_dimensions,
                                                          field_type=field_type)
        self.fields[name].latex_name = latex_name
        return self.fields[name]


class PyTorchDataHandling(MultiShapeDatahandling):

    class PyTorchArrayHandler:

        def __init__(self):
            pass

        def zeros(self, shape, dtype=np.float32, order='C'):
            assert order == 'C'

            return torch.zeros(*shape, dtype=numpy_dtype_to_torch(dtype))

        def ones(self, shape, dtype, order='C'):
            assert order == 'C'
            return torch.ones(*shape, dtype=numpy_dtype_to_torch(dtype))

        def empty(self, shape, dtype=np.float32, layout=None):
            if layout:
                cpu_array = torch.from_numpy(pystencils.field.create_numpy_array_with_layout(shape, dtype, layout))
                return self.from_numpy(cpu_array)
            else:
                return torch.empty(*shape, dtype=numpy_dtype_to_torch(dtype))

        def to_gpu(self, array):
            if not hasattr(array, 'cuda'):
                return torch.from_numpy(array).cuda()
            return array.cuda()

        def upload(self, gpuarray, numpy_array):
            if not hasattr(numpy_array, 'cuda'):
                numpy_array = torch.from_numpy(numpy_array)
            gpuarray[...] = numpy_array.cuda()

        def download(self, gpuarray, numpy_array):
            numpy_array[...] = gpuarray.cpu()

        def randn(self, shape, dtype=np.float32):
            return torch.randn(shape, dtype=dtype)

        from_numpy = torch.from_numpy

    def __init__(self,
                 domain_size: Sequence[int],
                 default_ghost_layers: int = 0,
                 default_layout: str = 'numpy',
                 periodicity: Union[bool, Sequence[bool]] = False,
                 default_target: str = 'gpu'):
        super().__init__(domain_size, default_ghost_layers, default_layout, periodicity, default_target)
        self.array_handler = self.PyTorchArrayHandler()

    def run_kernel(self, kernel_function, **kwargs):
        arrays = self.gpu_arrays if self.default_target == 'gpu' else self.cpu_arrays
        rtn = kernel_function(**arrays, **kwargs)
        return rtn

    def require_autograd(self, bool_val, *names):
        for n in names:
            try:
                self.cpu_arrays[n].require_autograd = bool_val
            except Exception:
                pass

            try:
                self.gpu_arrays[n].require_autograd = bool_val
            except Exception:
                pass
