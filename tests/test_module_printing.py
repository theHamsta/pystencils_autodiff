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
from pystencils_autodiff import create_backward_assignments
from pystencils_autodiff.backends.astnodes import PybindModule, TensorflowModule, TorchModule
from pystencils_autodiff.framework_integration.printer import FrameworkIntegrationPrinter

try:
    from pystencils.interpolation_astnodes import TextureCachedField
    HAS_INTERPOLATION = True
except ImportError:
    HAS_INTERPOLATION = False


def test_module_printing():
    module_name = "my_module"
    for target in ('cpu', 'gpu'):

        z, y, x = pystencils.fields("z, y, x: [2d]")

        forward_assignments = pystencils.AssignmentCollection({
            z[0, 0]: x[0, 0] * sympy.log(x[0, 0] * y[0, 0])
        })

        backward_assignments = create_backward_assignments(forward_assignments)

        forward_ast = pystencils.create_kernel(forward_assignments, target)
        forward_ast.function_name = 'forward'
        backward_ast = pystencils.create_kernel(backward_assignments, target)
        backward_ast.function_name = 'backward'
        module = TorchModule(module_name, [forward_ast, backward_ast])
        print(module)

        module = TensorflowModule(module_name, {forward_ast: backward_ast})
        print(module)

        if target == 'cpu':
            module = PybindModule(module_name, [forward_ast, backward_ast])
            print(module)
            module = PybindModule(module_name, forward_ast)
            print(module)


def test_module_printing_parameter():
    module_name = "Ololol"

    for target in ('cpu', 'gpu'):

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
        module = TorchModule(module_name, [forward_ast, backward_ast])
        print(module)

        module = TensorflowModule(module_name, {forward_ast: backward_ast})
        print(module)

        if target == 'cpu':
            module = PybindModule(module_name, [forward_ast, backward_ast])
            print(module)
            module = PybindModule(module_name, forward_ast)
            print(module)


@pytest.mark.skipif(not HAS_INTERPOLATION, reason="")
def test_module_printing_globals():
    for target in ('gpu',):

        z, y, x = pystencils.fields("z, y, x: [20,40]")

        forward_assignments = pystencils.AssignmentCollection({
            z[0, 0]: x[0, 0] * sympy.log(TextureCachedField(x).at(sympy.Matrix((0.43, 3))) * y[0, 0])
        })

        forward_ast = pystencils.create_kernel(forward_assignments, target)
        forward_ast.function_name = 'forward'
        module = TorchModule("hallo", [forward_ast])
        print(module)


def test_custom_printer():

    class DoesNotLikeTorchPrinter(FrameworkIntegrationPrinter):
        def _print_TorchModule(self, node):
            return 'Error: I don\'t like Torch'

    z, y, x = pystencils.fields("z, y, x: [20,40]")

    forward_assignments = pystencils.AssignmentCollection({
        z[0, 0]: x[0, 0] * sympy.log(TextureCachedField(x).at(sympy.Matrix((0.43, 3))) * y[0, 0])
    })

    forward_ast = pystencils.create_kernel(forward_assignments)
    forward_ast.function_name = 'forward'
    module = TorchModule("hallo", [forward_ast])
    print(DoesNotLikeTorchPrinter()(module))
