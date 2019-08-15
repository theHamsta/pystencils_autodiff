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
from pystencils_autodiff._io import _read_template_from_file

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


class JinjaCppFile(Node):
    TEMPLATE: jinja2.Template = None

    def __init__(self, ast_dict):
        self.ast_dict = ast_dict
        self.printer = MachineLearningBackend()
        Node.__init__(self)

    @property
    def args(self):
        """Returns all arguments/children of this node."""
        return self.ast_dict.values()

    @property
    def symbols_defined(self):
        """Set of symbols which are defined by this node."""
        return set()

    @property
    def undefined_symbols(self):
        """Symbols which are used but are not defined inside this node."""
        return set()

    def _print(self, node):
        if isinstance(node, Node):
            return self.printer(node)
        else:
            return str(node)

    def __str__(self):
        assert self.TEMPLATE, f"Template of {self.__class__} must be set"
        render_dict = {k: self._print(v) for k, v in self.ast_dict.items()}
        render_dict.update({"headers": get_headers(self)})

        return self.TEMPLATE.render(render_dict)

    def __repr__(self):
        return self.TEMPLATE.render(self.ast_dict)


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

    def _print_JinjaCppFile(self, node):
        return str(node)

    # def _print_DestructuringBindingsForFieldClass(self, node):
        # """
        # Can be deleted if supported by CBackend
        # """
        # # Define all undefined symbols
        # undefined_field_symbols = node.symbols_defined
        # destructuring_bindings = ["%s %s = %s.%s;" %
        # (u.dtype,
        # u.name,
        # u.field_name if hasattr(u, 'field_name') else u.field_names[0],
        # node.CLASS_TO_MEMBER_DICT[u.__class__].format(
        # dtype=(u.dtype.base_type if type(u) == FieldPointerSymbol else ""),
        # field_name=(u.field_name if hasattr(u, "field_name") else ""),
        # dim=("" if type(u) == FieldPointerSymbol else (u.coordinate,))
        # )
        # )
        # for u in undefined_field_symbols
        # ]
        # destructuring_bindings.sort()  # only for code aesthetics
        # return "{\n" + self._indent + \
        # ("\n" + self._indent).join(destructuring_bindings) + \
        # "\n" + self._indent + \
        # ("\n" + self._indent).join(self._print(node.body).splitlines()) + \
        # "\n}"
