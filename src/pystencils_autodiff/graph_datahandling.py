# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from enum import Enum

import numpy as np

import pystencils.datahandling
import pystencils.kernel_wrapper
import pystencils.timeloop
from pystencils.field import FieldType


class DataTransferKind(str, Enum):
    UNKNOWN = None
    HOST_ALLOC = 'HOST_ALLOC'
    DEVICE_ALLOC = 'DEVICE_ALLOC'
    HOST_TO_DEVICE = 'HOST_TO_DEVICE'
    DEVICE_TO_HOST = 'DEVICE_TO_HOST'
    HOST_COMMUNICATION = 'HOST_COMMUNICATION'
    DEVICE_COMMUNICATION = 'DEVICE_COMMUNICATION'
    HOST_SWAP = 'HOST_SWAP'
    DEVICE_SWAP = 'DEVICE_SWAP'
    HOST_GATHER = 'HOST_GATHER'
    DEVICE_GATHER = 'DEVICE_GATHER'

    def is_alloc(self):
        return self in [self.HOST_ALLOC, self.DEVICE_ALLOC]

    def is_transfer(self):
        return self in [self.HOST_TO_DEVICE, self.DEVICE_TO_HOST, self.SWAP]


class DataTransfer:
    def __init__(self, field: pystencils.Field, kind: DataTransferKind):
        self.field = field
        self.kind = kind

    def __str__(self):
        return f'DataTransferKind: {self.kind} with {self.field}'


class Swap(DataTransfer):
    def __init__(self, source, destination, gpu):
        self.kind = DataTransferKind.DEVICE_SWAP if gpu else DataTransferKind.HOST_SWAP
        self.field = source
        self.destination = destination

    def __str__(self):
        return f'Swap: {self.field} with {self.destination}'


class Communication(DataTransfer):
    def __init__(self, field, stencil, gpu):
        self.kind = DataTransferKind.DEVICE_COMMUNICATION if gpu else DataTransferKind.HOST_COMMUNICATION
        self.field = field
        self.stencil = stencil


class KernelCall:
    def __init__(self, kernel: pystencils.kernel_wrapper.KernelWrapper, kwargs):
        self.kernel = kernel
        self.kwargs = kwargs

    def __str__(self):
        return "Call " + str(self.kernel.ast.function_name)


class TimeloopRun:
    def __init__(self, timeloop, time_steps):
        self.timeloop = timeloop
        self.time_steps = time_steps

    def __str__(self):
        return (f'Timeloop:'
                + '\nPre:\n'
                + '\n   '.join(str(f) for f in self.timeloop._pre_run_functions)
                + f'\n{self.time_steps} time steps:\n'
                + '\n   '.join(str(f) for f in self.timeloop._single_step_asts)
                + '\nPost:\n'
                + '\n   '.join(str(f) for f in self.timeloop._post_run_functions))

    @property
    def asts(self):
        return self.timeloop._single_step_asts


class GraphDataHandling(pystencils.datahandling.SerialDataHandling):
    """Docstring for GraphDataHandling. """

    class TimeLoop(pystencils.timeloop.TimeLoop):
        def __init__(self, parent, *args, **kwargs):
            self.parent = parent
            self._single_step_asts = []
            super().__init__(*args, **kwargs)

        def add_pre_run_function(self, f):
            self._pre_run_functions.append(f)

        def add_post_run_function(self, f):
            self._post_run_functions.append(f)

        def add_single_step_function(self, f):
            self._single_step_functions.append(f)

        def add_call(self, functor, argument_list):
            for argument_dict in argument_list:
                self._single_step_asts.append((functor, argument_dict) if not hasattr(functor, 'ast') else functor.ast)

            if hasattr(functor, 'kernel'):
                functor = functor.kernel
            if not isinstance(argument_list, list):
                argument_list = [argument_list]

        def run(self, time_steps=1):
            self.parent.call_queue.append(TimeloopRun(self, time_steps))
            super().run(time_steps)

    def __init__(self, *args, **kwargs):

        self.call_queue = []
        super().__init__(*args, **kwargs)

    def add_array(self, name, values_per_cell=1, dtype=np.float64, latex_name=None, ghost_layers=None, layout=None,
                  cpu=True, gpu=None, alignment=False, field_type=FieldType.GENERIC):

        super().add_array(name,
                          values_per_cell,
                          dtype,
                          latex_name,
                          ghost_layers,
                          layout,
                          cpu,
                          gpu,
                          alignment,
                          field_type)
        if cpu:
            self.call_queue.append(DataTransfer(self._fields[name], DataTransferKind.HOST_ALLOC))
        if gpu:
            self.call_queue.append(DataTransfer(self._fields[name], DataTransferKind.DEVICE_ALLOC))

    def add_custom_data(self, name, cpu_creation_function,
                        gpu_creation_function=None, cpu_to_gpu_transfer_func=None, gpu_to_cpu_transfer_func=None):

        self.call_queue.append('custom data. WTF?')
        super().add_custom_data(name, cpu_creation_function,
                                gpu_creation_function, cpu_to_gpu_transfer_func, gpu_to_cpu_transfer_func)

    def gather_array(self, name, slice_obj=None, ghost_layers=False, **kwargs):

        self.call_queue.append('gather_array')
        super().gather_array(name, slice_obj, ghost_layers, **kwargs)

    def swap(self, name1, name2, gpu=None):

        self.call_queue.append(Swap(self._fields[name1], self._fields[name2], gpu))
        super().swap(name1, name2, gpu)

    def run_kernel(self, kernel_function, **kwargs):
        self.call_queue.append(KernelCall(kernel_function, kwargs))
        super().run_kernel(kernel_function, **kwargs)
        # skip calling super

    def to_cpu(self, name):
        super().to_cpu(name)
        self.call_queue.append(DataTransfer(self._fields[name], DataTransferKind.HOST_TO_DEVICE))

    def to_gpu(self, name):
        super().to_gpu(name)
        if name in self._custom_data_transfer_functions:
            self.call_queue.append('Custom Tranfer Function')
        else:
            self.call_queue.append(DataTransfer(self._fields[name], DataTransferKind.DEVICE_TO_HOST))

    def synchronization_function(self, names, stencil=None, target=None, **_):
        for name in names:
            gpu = target == 'gpu'
            self.call_queue.append(Communication(self._fields[name], stencil, gpu))
        super().synchronization_function(names, stencil=None, target=None, **_)

    def __str__(self):
        return '\n'.join(str(c) for c in self.call_queue)

    def create_timeloop(self, *args, **kwargs):
        return self.TimeLoop(self, *args, **kwargs)

    def fill(self, array_name: str, val, value_idx=None,
             slice_obj=None, ghost_layers=False, inner_ghost_layers=False) -> None:
        self.call_queue.append('Fill ' + array_name)
        super().fill(array_name, val, value_idx, slice_obj, ghost_layers, inner_ghost_layers)

    # TODO
    # def reduce_float_sequence(self, sequence, operation, all_reduce=False) -> np.array:
        # return np.array(sequence)

    # def reduce_int_sequence(self, sequence, operation, all_reduce=False) -> np.array:
        # return np.array(sequence)

    # def create_vtk_writer(self, file_name, data_names, ghost_layers=False):
        # pass

    # def create_vtk_writer_for_flag_array(self, file_name, data_name, masks_to_name, ghost_layers=False):
        # pass
