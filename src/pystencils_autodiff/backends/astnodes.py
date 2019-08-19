# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

from collections.abc import Iterable
from os.path import dirname, join

import jinja2

from pystencils.astnodes import (
    DestructuringBindingsForFieldClass, FieldPointerSymbol, FieldShapeSymbol,
    FieldStrideSymbol, Node)
from pystencils_autodiff._file_io import _read_template_from_file
from pystencils_autodiff.framework_integration.astnodes import (
    JinjaCppFile, WrapperFunction, generate_kernel_call)

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
        FieldPointerSymbol: "flat<{dtype}>()",
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
            'python_bindings': PybindPythonBindings(module_name, [PybindFunctionWrapping(a) for a in wrapper_functions])
        }

        super().__init__(ast_dict)

    def generate_wrapper_function(self, kernel_ast):
        return WrapperFunction(self.DESTRUCTURING_CLASS(generate_kernel_call(kernel_ast)),
                               function_name='call_' + kernel_ast.function_name)


class TensorflowModule(TorchModule):
    DESTRUCTURING_CLASS = TensorflowTensorDestructuring

    def __init__(self, module_name, forward_to_backward_kernel_dict):
        """Create a C++ module with forward and optional backward_kernels

        :param forward_kernel_ast: one or more kernel ASTs (can have any C dialect)
        :param backward_kernel_ast:
        """

        self._forward_to_backward_dict = forward_to_backward_kernel_dict
        kernel_asts = list(forward_to_backward_kernel_dict.values()) + list(forward_to_backward_kernel_dict.keys())
        super().__init__(module_name, kernel_asts)


class PybindModule(TorchModule):
    DESTRUCTURING_CLASS = PybindArrayDestructuring


class PybindPythonBindings(JinjaCppFile):
    TEMPLATE = jinja2.Template("""PYBIND11_MODULE("{{ module_name }}", m)
{
{% for ast_node in module_contents -%}
{{ ast_node | indent(3,true) }}
{% endfor -%}
}
""")

    def __init__(self, module_name, astnodes_to_wrap):
        super().__init__({'module_name': module_name, 'module_contents': astnodes_to_wrap})


class PybindFunctionWrapping(JinjaCppFile):
    TEMPLATE = jinja2.Template(
        """m.def("{{ python_name }}", &{{ cpp_name }}, {% for p in parameters -%}"{{ p }}"_a{{- ", " if not loop.last -}}{% endfor %});"""  # noqa
    )

    required_global_declarations = ["using namespace pybind11::literals;"]
    headers = ['<pybind11/pybind11.h>',
               '<pybind11/stl.h>']

    def __init__(self, function_node):
        super().__init__({'python_name': function_node.function_name,
                          'cpp_name': function_node.function_name,
                          'parameters': [p.symbol.name for p in function_node.get_parameters()]
                          })
