# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""


import itertools

import numpy as np
import pytest
import sympy as sp

import pystencils
from pystencils_autodiff.transformations import add_fixed_constant_boundary_handling


@pytest.mark.parametrize('num_ghost_layers', (1, 2, 3))
def test_fixed_constant_bh(num_ghost_layers):
    ndim = 2

    offsets = list(itertools.product(range(num_ghost_layers + 1), repeat=ndim))

    x, y = pystencils.fields(f'x, y:  float64[{ndim}d]')

    assignments = pystencils.AssignmentCollection({
        y.center: sp.Add(*[x.__getitem__(o) for o in offsets]) / len(offsets)

    })

    kernel = pystencils.create_kernel(assignments).compile()
    print(kernel.code)

    bh_assignments = add_fixed_constant_boundary_handling(assignments, num_ghost_layers)

    bh_kernel = pystencils.create_kernel(bh_assignments, ghost_layers=0).compile()
    print(bh_kernel.code)

    noise = np.random.rand(*[20, 30, 40][:ndim])
    out1 = np.zeros_like(noise)
    out2 = np.zeros_like(noise)

    kernel(x=noise, y=out1)
    bh_kernel(x=noise, y=out2)
