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
from typing import Sequence, Union

import numpy as np

import pystencils
from pystencils.autodiff.backends._pytorch import torch_dtype_to_numpy
from pystencils.field import (
    Field, FieldType, create_numpy_array_with_layout, layout_string_to_tuple,
    spatial_layout_string_to_tuple)
from pystencils_autodiff.field_tensor_conversion import _torch_tensor_to_numpy_shim


class PyTorchDataHandling(pystencils.datahandling.SerialDataHandling):

    class PyTorchArrayHandler:

        def __init__(self):
            pass

        def zeros(self, shape, dtype=np.float32, order='C'):
            assert order == 'C'

            return torch.zeros(*shape, dtype=torch_dtype_to_numpy(dtype))

        def ones(self, shape, dtype, order='C'):
            assert order == 'C'
            return torch.ones(*shape, dtype=torch_dtype_to_numpy(dtype))

        def empty(self, shape, dtype=np.float32, layout=None):
            if layout:
                cpu_array = torch.from_numpy(pystencils.field.create_numpy_array_with_layout(shape, dtype, layout))
                return self.from_numpy(cpu_array)
            else:
                return torch.empty(*shape, dtype=torch_dtype_to_numpy(dtype))

        def to_gpu(self, array):
            return array.cuda()

        def upload(self, gpuarray, numpy_array):
            gpuarray[...] = numpy_array.cuda()

        def download(self, gpuarray, numpy_array):
            numpy_array[...] = gpuarray.cpu()

        def randn(self, shape, dtype=np.float64):
            cpu_array = torch.from_numpy(np.random.randn(*shape).astype(dtype))
            return self.from_numpy(cpu_array)

    def __init__(self,
                 domain_size: Sequence[int],
                 default_ghost_layers: int = 0,
                 default_layout: str = 'SoA',
                 periodicity: Union[bool, Sequence[bool]] = False,
                 default_target: str = 'gpu'):
        super().__init__(domain_size, default_ghost_layers, default_layout, periodicity, default_target)
        self.array_handler = self.PyTorchArrayHandler()

    def add_array(self, name, values_per_cell=1, dtype=np.float64, latex_name=None, ghost_layers=None, layout=None,
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
        cpu_arr = torch.from_numpy(create_numpy_array_with_layout(layout=layout_tuple, alignment=alignment,
                                                                  byte_offset=byte_offset, **kwargs))

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
                                                          _torch_tensor_to_numpy_shim(cpu_arr),
                                                          index_dimensions=index_dimensions,
                                                          field_type=field_type)
        self.fields[name].latex_name = latex_name
        return self.fields[name]
