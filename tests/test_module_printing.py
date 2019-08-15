# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import sympy

import pystencils
from pystencils_autodiff import create_backward_assignments
from pystencils_autodiff.backends.astnodes import TensorCudaModule, TorchCudaModule

TARGET_TO_DIALECT = {
    'cpu': 'c',
    'gpu': 'cuda'
}


def test_module_printing():
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
        module = TorchCudaModule(forward_ast, backward_ast)
        print(module)

        module = TensorCudaModule(forward_ast, backward_ast)
        print(module)


def test_module_printing_parameter():
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
        module = TorchCudaModule(forward_ast, backward_ast)
        print(module)

        module = TensorCudaModule(forward_ast, backward_ast)
        print(module)


def main():
    test_module_printing()
    test_module_printing_parameter()


if __name__ == '__main__':
    main()
