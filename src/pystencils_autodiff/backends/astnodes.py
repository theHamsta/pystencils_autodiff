# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

from collections.abc import Iterable
from os.path import dirname, join

from pystencils.astnodes import FieldPointerSymbol, FieldShapeSymbol, FieldStrideSymbol
from pystencils_autodiff._file_io import _read_template_from_file
from pystencils_autodiff.backends.python_bindings import (
    PybindFunctionWrapping, PybindPythonBindings, TensorflowFunctionWrapping,
    TensorflowPythonBindings, TorchPythonBindings)
from pystencils_autodiff.framework_integration.astnodes import (
    DestructuringBindingsForFieldClass, JinjaCppFile, WrapperFunction, generate_kernel_call)

# Torch


class TorchTensorDestructuring(DestructuringBindingsForFieldClass):
    CLASS_TO_MEMBER_DICT = {
        FieldPointerSymbol: "data<{dtype}>()",
        FieldShapeSymbol: "size({dim})",
        FieldStrideSymbol: "strides()[{dim}]"
    }

    CLASS_NAME_TEMPLATE = "at::Tensor"

    headers = ["<ATen/ATen.h>"]


class TensorflowTensorDestructuring(DestructuringBindingsForFieldClass):
    CLASS_TO_MEMBER_DICT = {
        FieldPointerSymbol: "flat<{dtype}>().data()",
        FieldShapeSymbol: "dim_size({dim})",
        FieldStrideSymbol: "dim_size({dim}) * tensorflow::DataTypeSize({field_name}.dtype())"
    }

    CLASS_NAME_TEMPLATE = "tensorflow::Tensor"

    headers = ["<tensorflow/core/framework/tensor.h>",
               "<tensorflow/core/framework/types.h>",
               ]


class PybindArrayDestructuring(DestructuringBindingsForFieldClass):
    CLASS_TO_MEMBER_DICT = {
        FieldPointerSymbol: "mutable_data()",
        FieldShapeSymbol: "shape({dim})",
        FieldStrideSymbol: "strides({dim})"
    }

    CLASS_NAME_TEMPLATE = "pybind11::array_t<{dtype}>"

    headers = ["<pybind11/numpy.h>"]


class TorchModule(JinjaCppFile):
    TEMPLATE = _read_template_from_file(join(dirname(__file__), 'module.tmpl.cpp'))
    DESTRUCTURING_CLASS = TorchTensorDestructuring
    PYTHON_BINDINGS_CLASS = TorchPythonBindings
    PYTHON_FUNCTION_WRAPPING_CLASS = PybindFunctionWrapping

    def __init__(self, module_name, kernel_asts):
        """Create a C++ module with forward and optional backward_kernels

        :param forward_kernel_ast: one or more kernel ASTs (can have any C dialect)
        :param backward_kernel_ast:
        """
        if not isinstance(kernel_asts, Iterable):
            kernel_asts = [kernel_asts]
        wrapper_functions = [self.generate_wrapper_function(k) for k in kernel_asts]

        ast_dict = {
            'kernels': kernel_asts,
            'kernel_wrappers': wrapper_functions,
            'python_bindings': self.PYTHON_BINDINGS_CLASS(module_name,
                                                          [self.PYTHON_FUNCTION_WRAPPING_CLASS(a)
                                                              for a in wrapper_functions])
        }

        super().__init__(ast_dict)

    def generate_wrapper_function(self, kernel_ast):
        return WrapperFunction(self.DESTRUCTURING_CLASS(generate_kernel_call(kernel_ast)),
                               function_name='call_' + kernel_ast.function_name)


class TensorflowModule(TorchModule):
    DESTRUCTURING_CLASS = TensorflowTensorDestructuring
    PYTHON_BINDINGS_CLASS = TensorflowPythonBindings
    PYTHON_FUNCTION_WRAPPING_CLASS = TensorflowFunctionWrapping

    def __init__(self, module_name, kernel_asts, use_cuda=False):
        """Create a C++ module with forward and optional backward_kernels

        :param forward_kernel_ast: one or more kernel ASTs (can have any C dialect)
        :param backward_kernel_ast:
        """
        if use_cuda:
            self.TEMPLATE = _read_template_from_file(join(dirname(__file__), 'tensorflow.cuda.tmpl.cu'))

        super().__init__(module_name, kernel_asts)


class PybindModule(TorchModule):
    DESTRUCTURING_CLASS = PybindArrayDestructuring
    PYTHON_BINDINGS_CLASS = PybindPythonBindings
