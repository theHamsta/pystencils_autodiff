# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pytest
import sympy as sp

import pystencils
from pystencils.astnodes import Block
from pystencils_autodiff.framework_integration.astnodes import (
    DestructuringBindingsForFieldClass, FunctionCall, WrapperFunction, generate_kernel_call)
from pystencils_autodiff.framework_integration.printer import FrameworkIntegrationPrinter

# TODO
try:
    from pystencils.interpolation_astnodes import LinearInterpolator
    HAS_INTERPOLATION = True
except ImportError:
    HAS_INTERPOLATION = False


def test_pure_call():

    z, y, x = pystencils.fields("z, y, x: [2d]")

    forward_assignments = pystencils.AssignmentCollection([pystencils.Assignment(
        z[0, 0], x[0, 0] * sp.log(x[0, 0] * y[0, 0]))], [])

    for target in ('cpu', 'gpu'):
        ast = pystencils.create_kernel(forward_assignments, target=target)
        kernel_call_ast = FunctionCall(ast)
        code = FrameworkIntegrationPrinter()(kernel_call_ast)
        print(code)


def test_call_with_destructuring():
    z, y, x = pystencils.fields("z, y, x: [2d]")

    forward_assignments = pystencils.AssignmentCollection([pystencils.Assignment(
        z[0, 0], x[0, 0] * sp.log(x[0, 0] * y[0, 0]))], [])

    for target in ('cpu', 'gpu'):
        ast = pystencils.create_kernel(forward_assignments, target=target)
        kernel_call_ast = FunctionCall(ast)
        wrapper = DestructuringBindingsForFieldClass(kernel_call_ast)
        code = FrameworkIntegrationPrinter()(wrapper)
        print(code)


def test_call_with_destructuring_fixed_size():

    z, y, x = pystencils.fields("z, y, x: [100, 80]")

    forward_assignments = pystencils.AssignmentCollection([pystencils.Assignment(
        z[0, 0], x[0, 0] * sp.log(x[0, 0] * y[0, 0]))], [])

    for target in ('cpu', 'gpu'):
        ast = pystencils.create_kernel(forward_assignments, target=target)
        kernel_call_ast = FunctionCall(ast)
        wrapper = DestructuringBindingsForFieldClass(kernel_call_ast)
        code = FrameworkIntegrationPrinter()(wrapper)
        print(code)


def test_wrapper_function():
    z, y, x = pystencils.fields("z, y, x: [100, 80]")

    forward_assignments = pystencils.AssignmentCollection([pystencils.Assignment(
        z[0, 0], x[0, 0] * sp.log(x[0, 0] * y[0, 0]))], [])

    for target in ('cpu', 'gpu'):
        ast = pystencils.create_kernel(forward_assignments, target=target)
        kernel_call_ast = FunctionCall(ast)
        wrapper = WrapperFunction(DestructuringBindingsForFieldClass(kernel_call_ast))
        code = FrameworkIntegrationPrinter()(wrapper)
        print(code)

    for target in ('cpu', 'gpu'):
        ast = pystencils.create_kernel(forward_assignments, target=target)
        kernel_call_ast = FunctionCall(ast)
        wrapper = WrapperFunction(Block([kernel_call_ast]))
        code = FrameworkIntegrationPrinter()(wrapper)
        print(code)


@pytest.mark.skipif(not HAS_INTERPOLATION, reason="")
@pytest.mark.parametrize('target', ('cpu', 'gpu'))
@pytest.mark.parametrize('with_wrapper', ('with_wrapper', False))
def test_generate_kernel_call(target, with_wrapper):
    z, y, x = pystencils.fields("z, y, x: [100, 80]")

    forward_assignments = pystencils.AssignmentCollection([pystencils.Assignment(
        z[0, 0], LinearInterpolator(x).at((0, 0)) * sp.log(x[0, 0] * y[0, 0]))])

    if not with_wrapper:
        ast = pystencils.create_kernel(forward_assignments, target=target)
        kernel_call_ast = generate_kernel_call(ast)
        code = FrameworkIntegrationPrinter()(kernel_call_ast)
        print(code)
    else:
        ast = pystencils.create_kernel(forward_assignments, target=target)
        kernel_call_ast = generate_kernel_call(ast)
        wrapper = WrapperFunction(DestructuringBindingsForFieldClass(kernel_call_ast))
        code = FrameworkIntegrationPrinter()(wrapper)
        print(code)


@pytest.mark.skipif(not HAS_INTERPOLATION, reason="")
def test_generate_kernel_call_only_texture_for_one_field_fixed_size():
    z, y, x = pystencils.fields("z, y, x: [100, 80]")

    forward_assignments = pystencils.AssignmentCollection([pystencils.Assignment(
        z[0, 0], LinearInterpolator(x).at((0, 0)) * sp.log(LinearInterpolator(x).at((0, 0)) * y[0, 0]))])

    for target in ('cpu', 'gpu'):
        ast = pystencils.create_kernel(forward_assignments, target=target)
        kernel_call_ast = generate_kernel_call(ast)
        code = FrameworkIntegrationPrinter()(kernel_call_ast)
        print(code)

    for target in ('cpu', 'gpu'):
        ast = pystencils.create_kernel(forward_assignments, target=target)
        kernel_call_ast = generate_kernel_call(ast)
        wrapper = WrapperFunction(DestructuringBindingsForFieldClass(kernel_call_ast))
        code = FrameworkIntegrationPrinter()(wrapper)
        print(code)


@pytest.mark.skipif(not HAS_INTERPOLATION, reason="")
def test_generate_kernel_call_only_texture_for_one_field():

    z, y, x = pystencils.fields("z, y, x")

    forward_assignments = pystencils.AssignmentCollection([pystencils.Assignment(
        z[0, 0], LinearInterpolator(x).at((0, 0)) * sp.log(LinearInterpolator(x).at((0, 0)) * y[0, 0]))])

    for target in ('cpu', 'gpu'):
        ast = pystencils.create_kernel(forward_assignments, target=target)
        kernel_call_ast = generate_kernel_call(ast)
        code = FrameworkIntegrationPrinter()(kernel_call_ast)
        print(code)

    for target in ('cpu', 'gpu'):
        ast = pystencils.create_kernel(forward_assignments, target=target)
        kernel_call_ast = generate_kernel_call(ast)
        wrapper = WrapperFunction(DestructuringBindingsForFieldClass(kernel_call_ast))
        code = FrameworkIntegrationPrinter()(wrapper)
        print(code)
