# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

from os.path import dirname, join

import jinja2

from pystencils.astnodes import (
    DestructuringBindingsForFieldClass, FieldPointerSymbol, FieldShapeSymbol,
    FieldStrideSymbol, Node)
from pystencils.backends.cbackend import get_headers
from pystencils.framework_intergration_astnodes import (
    FrameworkIntegrationPrinter, WrapperFunction, generate_kernel_call)
from pystencils_autodiff._file_io import _read_template_from_file
from pystencils_autodiff.framework_integration.astnodes import JinjaCppFile

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


class TorchCudaModule(JinjaCppFile):
    TEMPLATE = _read_template_from_file(join(dirname(__file__), 'torch_native_cuda.tmpl.cu'))
    DESTRUCTURING_CLASS = TorchTensorDestructuring

    def __init__(self, forward_kernel_ast, backward_kernel_ast):

        ast_dict = {
            'forward_kernel': forward_kernel_ast,
            'backward_kernel': backward_kernel_ast,
            'forward_wrapper': self.generate_wrapper_function(forward_kernel_ast),
            'backward_wrapper': self.generate_wrapper_function(backward_kernel_ast),
        }

        super().__init__(ast_dict)

    def generate_wrapper_function(self, kernel_ast):
        return WrapperFunction(self.DESTRUCTURING_CLASS(generate_kernel_call(kernel_ast)),
                               function_name='call_' + kernel_ast.function_name)


class TensorCudaModule(TorchCudaModule):
    DESTRUCTURING_CLASS = TensorflowTensorDestructuring


# Generic
class MachineLearningBackend(FrameworkIntegrationPrinter):
