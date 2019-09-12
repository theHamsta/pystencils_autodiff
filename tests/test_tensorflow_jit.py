# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import pytest
import sympy

import pystencils
import pystencils_autodiff
from pystencils_autodiff import create_backward_assignments
from pystencils_autodiff.backends.astnodes import TensorflowModule


def test_tensorflow_jit_cpu():

    pytest.importorskip('tensorflow')

    module_name = "Ololol"

    target = 'cpu'

    z, y, x = pystencils.fields("z, y, x: [20,40]")
    a = sympy.Symbol('a')

    forward_assignments = pystencils.AssignmentCollection({
        z[0, 0]: x[0, 0] * sympy.log(a * x[0, 0] * y[0, 0])
    })

    backward_assignments = create_backward_assignments(forward_assignments)

    forward_ast = pystencils.create_kernel(forward_assignments, target)
    forward_ast.function_name = 'forward'
    backward_ast = pystencils.create_kernel(backward_assignments, target)
    backward_ast.function_name = 'backward'
    module = TensorflowModule(module_name, [forward_ast, backward_ast])
    print(module)

    lib = pystencils_autodiff.tensorflow_jit.compile_sources_and_load([str(module)])
    assert 'call_forward' in dir(lib)
    assert 'call_backward' in dir(lib)
