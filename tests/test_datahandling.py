# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pytest
import sympy

import pystencils
from pystencils_autodiff.framework_integration.datahandling import PyTorchDataHandling

pystencils_reco = pytest.importorskip('pystencils_reco')


def test_datahandling():
    dh = PyTorchDataHandling((20, 30))

    dh.add_array('x')
    dh.add_array('y')
    dh.add_array('z')
    a = sympy.Symbol('a')

    z, y, x = pystencils.fields("z, y, x: [20,40]")
    forward_assignments = pystencils_reco.AssignmentCollection({
        z[0, 0]: x[0, 0] * sympy.log(a * x[0, 0] * y[0, 0])
    })

    kernel = forward_assignments.create_pytorch_op()

    dh.run_kernel(kernel, a=3)
