# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from copy import copy
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


# class BoundaryHandling:
    # def __init__(self, field: pystencils.Field):
        # self.field = field

    # def __str__(self):
        # return f'BoundaryHandling on {self.field}'

    # def __repr__(self):
        # return self.__str__()


class DataTransfer:
    def __init__(self, field: pystencils.Field, kind: DataTransferKind):
        self.field = field
        self.kind = kind

    def __str__(self):
        return f'DataTransferKind: {self.kind} with {self.field}'

    def __repr__(self):
        return f'DataTransferKind: {self.kind} with {self.field}'


class Swap(DataTransfer):
    def __init__(self, source, destination, gpu):
        self.kind = DataTransferKind.DEVICE_SWAP if gpu else DataTransferKind.HOST_SWAP
        self.field = source
        self.destination = destination

    def __repr__(self):
        return f'Swap: {self.field} with {self.destination}'


class Communication(DataTransfer):
    def __init__(self, field, stencil, gpu):
        self.kind = DataTransferKind.DEVICE_COMMUNICATION if gpu else DataTransferKind.HOST_COMMUNICATION
        self.field = field
        self.stencil = stencil


class KernelCall:
    def __init__(self, kernel: pystencils.kernel_wrapper.KernelWrapper, kwargs, tmp_field_swaps=[]):
        self.kernel = kernel
        self.kwargs = kwargs
        self.tmp_field_swaps = tmp_field_swaps

    def __str__(self):
        return "Call " + str(self.kernel.ast.function_name)


class FieldOutput:
    def __init__(self, fields, output_path, flag_field):
        self.fields = fields
        self.output_path = output_path
        self.flag_field = flag_field

    def __str__(self):
        return "Writing fields " + str(self.fields)


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
        return [s.kernel.ast for s in self.timeloop._single_step_asts if hasattr(s, 'kernel')]


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
            if hasattr(functor, 'ast'):
                self._single_step_asts.append(KernelCall(functor, {}))
            else:
                former_queue = self.parent.call_queue
                self.parent.call_queue = []
                #
                functor(*argument_list)
                self._single_step_asts.extend(self.parent.call_queue)
                self.parent.call_queue = former_queue

        def run(self, time_steps=1):
            former_call_queue = copy(self.parent.call_queue)
            self.parent.call_queue = []
            super().run(time_steps)
            self.parent.call_queue = former_call_queue
            former_call_queue.append(TimeloopRun(self, time_steps))

        def swap(self, src, dst, is_gpu):
            if isinstance(src, str):
                src = self.parent.fields[src]
            if isinstance(dst, str):
                dst = self.parent.fields[dst]
            self._single_step_asts.append(Swap(src, dst, is_gpu))

    def __init__(self, *args, **kwargs):

        self.call_queue = []
        self._timeloop_record = None
        super().__init__(*args, **kwargs)

    def add_array(self,
                  name,
                  values_per_cell=1,
                  dtype=np.float64,
                  latex_name=None,
                  ghost_layers=None,
                  layout=None,
                  cpu=True,
                  gpu=None,
                  alignment=False,
                  field_type=FieldType.GENERIC):

        rtn = super().add_array(name,
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
        return rtn

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

    def run_kernel(self, kernel_function, simulate_only=False, **kwargs):
        self.call_queue.append(KernelCall(kernel_function, kwargs))
        if not simulate_only:
            super().run_kernel(kernel_function, **kwargs)

    def to_cpu(self, name):
        super().to_cpu(name)
        self.call_queue.append(DataTransfer(self._fields[name], DataTransferKind.DEVICE_TO_HOST))

    def to_gpu(self, name):
        super().to_gpu(name)
        if name in self._custom_data_transfer_functions:
            self.call_queue.append('Custom Tranfer Function')
        else:
            self.call_queue.append(DataTransfer(self._fields[name], DataTransferKind.HOST_TO_DEVICE))

    def synchronization_function(self, names, stencil=None, target=None, **_):
        def func():
            for name in names:
                gpu = target == 'gpu'
                self.call_queue.append(Communication(self._fields[name], stencil, gpu))
            pystencils.datahandling.SerialDataHandling.synchronization_function(self,
                                                                                names,
                                                                                stencil=None,
                                                                                target=None,
                                                                                **_)

        return func

    def __str__(self):
        return '\n'.join(str(c) for c in self.call_queue)

    def create_timeloop(self, *args, **kwargs):
        return self.TimeLoop(self, *args, **kwargs)

    def fill(self, array_name: str, val, value_idx=None,
             slice_obj=None, ghost_layers=False, inner_ghost_layers=False) -> None:
        self.call_queue.append('Fill ' + array_name)
        super().fill(array_name, val, value_idx, slice_obj, ghost_layers, inner_ghost_layers)

    def merge_swaps_with_kernel_calls(self, call_queue=None):
        # TODO(seitz): should be moved to ComputationGraph

        if call_queue is None:
            call_queue = self.call_queue

        relevant_swaps = [(swap, predecessor) for (swap, predecessor) in zip(call_queue[1:], call_queue[:-1])
                          if isinstance(swap, Swap) and isinstance(predecessor, KernelCall)]
        for s, pred in relevant_swaps:
            call_queue.remove(s)
            if (s.field, s.destination) not in pred.tmp_field_swaps:
                pred.tmp_field_swaps.append((s.field, s.destination))

        for t in call_queue:
            if isinstance(t, TimeloopRun):
                self.merge_swaps_with_kernel_calls(t.timeloop._single_step_asts)

    def save_fields(self, fields, output_path, flag_field=None):
        if isinstance(fields, str):
            fields = [fields]
        fields = [self.fields[f] if isinstance(f, str) else f for f in fields]
        self.call_queue.append(FieldOutput(fields, output_path, flag_field))

    # TODO
    # def reduce_float_sequence(self, sequence, operation, all_reduce=False) -> np.array:
        # return np.array(sequence)

    # def reduce_int_sequence(self, sequence, operation, all_reduce=False) -> np.array:
        # return np.array(sequence)

    # def create_vtk_writer(self, file_name, data_names, ghost_layers=False):
        # pass

    # def create_vtk_writer_for_flag_array(self, file_name, data_name, masks_to_name, ghost_layers=False):
        # pass
