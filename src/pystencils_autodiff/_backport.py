# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import itertools

import pystencils
from pystencils.astnodes import KernelFunction, ResolvedFieldAccess, SympyAssignment
from pystencils.interpolation_astnodes import InterpolatorAccess


def compatibility_hacks():

    def fields_written(self):
        assignments = self.atoms(SympyAssignment)
        return {a.lhs.field for a in assignments if isinstance(a.lhs, ResolvedFieldAccess)}

    def fields_read(self):
        assignments = self.atoms(SympyAssignment)
        return set().union(itertools.chain.from_iterable([f.field for f in a.rhs.free_symbols
                                                          | a.rhs.atoms(InterpolatorAccess) if hasattr(f, 'field')]
                                                         for a in assignments))

    _pystencils_fields = pystencils.fields

    def fields(*args, **kwargs):
        try:
            import torch
            from pystencils_autodiff.field_tensor_conversion import _torch_tensor_to_numpy_shim
            kwargs = {k: _torch_tensor_to_numpy_shim(v) if isinstance(
                v, torch.Tensor) else v for k, v in kwargs.items()}
        except ImportError:
            torch = None
        return _pystencils_fields(*args, **kwargs)

    pystencils.fields = fields
    KernelFunction.fields_read = property(fields_read)
    KernelFunction.fields_written = property(fields_written)


compatibility_hacks()
