# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import jinja2
import stringcase

from pystencils.astnodes import KernelFunction
from pystencils_autodiff.framework_integration.astnodes import JinjaCppFile


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


class TorchPythonBindings(JinjaCppFile):
    TEMPLATE = jinja2.Template("""PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
{% for ast_node in module_contents -%}
{{ ast_node | indent(3,true) }}
{% endfor -%}
}
""")
    headers = ['<torch/extension.h>']

    def __init__(self, module_name, astnodes_to_wrap):
        super().__init__({'module_contents': astnodes_to_wrap})


class TensorflowPythonBindings(JinjaCppFile):
    TEMPLATE = jinja2.Template(
        """
{% for b in bindings -%}
{{ b }}
{% endfor -%}
"""  # noqa
    )

    def __init__(self, module_name, python_bindings):
        super().__init__({'bindings': python_bindings})


class TensorflowFunctionWrapping(JinjaCppFile):
    TEMPLATE = jinja2.Template(
        """
REGISTER_OP("{{ python_name }}")
    {% for f in input_fields -%}
    .Input("{{ f.name }}: {{ f.dtype }}")
    {% endfor -%}
    {% for f in output_fields -%}
    .Output("{{ f.name }}: {{ f.dtype }}")
    {% endfor -%}
    {% for s in other_symbols -%}
    .Input("{{ s.name }}: {{ s.dtype }}")
    {% endfor -%}
    .Doc(
R"doc(
{{ docstring }}
  )doc");


class {{ python_name }} : public OpKernel {
public:
    explicit {{ python_name }}(OpKernelConstruction* context) : OpKernel(context)
    {
         {{ constructor }}
    }

    auto Compute(OpKernelContext* context) override -> void
    {
        {{ compute_method }}
    }
};

REGISTER_KERNEL_BUILDER(Name("{{ python_name }}").Device({{ device }}), {{ python_name }});
        """  # noqa
    )

    required_global_declarations = ["using namespace tensorflow;"]
    headers = ['"tensorflow/core/framework/op.h"',
               '"tensorflow/core/framework/op_kernel.h"']

    def __init__(self, function_node: KernelFunction):
        input_fields = list(function_node.fields_read)
        output_fields = list(function_node.fields_written)
        input_field_names = [f.name for f in input_fields]
        output_field_names = [f.name for f in output_fields]
        parameters = function_node.get_parameters()

        docstring = "TODO"  # TODO

        # this looks almost like lisp 😕
        other_symbols = [p.symbol
                         for p in parameters
                         if (p.symbol.name not in input_field_names and
                             p.symbol.name not in output_field_names)]

        # TODO dtype -> tf dtype mapping

        super().__init__({'python_name': stringcase.pascalcase(function_node.function_name),  # tf wants PascalCase!
                          'cpp_name': function_node.function_name,
                          'parameters': [p.symbol.name for p in parameters],
                          'input_fields': input_fields,
                          'output_fields': output_fields,
                          'other_symbols': other_symbols,
                          'docstring': docstring,
                          'device': 'DEVICE_GPU' if function_node.backend == 'gpucuda' else 'DEVICE_CPU',
                          })


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
