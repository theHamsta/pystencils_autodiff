# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import itertools

import jinja2
import stringcase

from pystencils.astnodes import KernelFunction, Node
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
    {% for f in inputs -%}
    .Input("{{ f.name }}: {{ f.dtype }}")
    {% endfor -%}
    {% for f in output_fields -%}
    .Output("{{ f.name }}: {{ f.dtype }}")
    {% endfor -%}
    .Doc(
R"doc(
{{ docstring }}
  )doc");


class {{ python_name }} : public OpKernel {
public:
    explicit {{ python_name }}(OpKernelConstruction* context) : OpKernel(context)
    {
{{ constructor | indent(8,true) }}
    }

    void Compute(OpKernelContext* context) override
    {
{{ compute_method | indent(8,true) }}
    }
};

REGISTER_KERNEL_BUILDER(Name("{{ python_name }}").Device({{ device }}), {{ python_name }});
        """  # noqa
    )

    _required_global_declarations = ["using namespace tensorflow;"]
    _headers = ['"tensorflow/core/framework/op.h"',
                '"tensorflow/core/framework/op_kernel.h"']

    @property
    def headers(self):
        return self._headers  # + ['"tensorflow/core/util/gpu_kernel_helper.h"'] if self.is_cuda else self._headers

    @property
    def required_global_declarations(self):
        return self._required_global_declarations  # + ['#define EIGEN_USE_GPU'] \
        # if self.is_cuda else self._required_global_declarations

    def __init__(self, function_node: KernelFunction):
        input_fields = list(function_node.fields_read)
        output_fields = list(function_node.fields_written)
        input_field_names = [f.name for f in input_fields]
        output_field_names = [f.name for f in output_fields]
        parameters = function_node.get_parameters()
        output_shape = str(output_fields[0].shape).replace('(', '{').replace(')', '}')  # noqa,  TODO make work for flexible sizes

        print([f for f in function_node.atoms(Node)])

        docstring = "TODO"  # TODO

        # this looks almost like lisp 😕
        other_symbols = [p.symbol
                         for p in parameters
                         if (p.symbol.name not in input_field_names and
                             p.symbol.name not in output_field_names)]
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.other_symbols = other_symbols
        self.is_cuda = any(f.backend == 'gpucuda' for f in function_node.atoms(KernelFunction))

        render_dict = {'python_name': stringcase.pascalcase(function_node.function_name),  # tf wants PascalCase!
                       'cpp_name': function_node.function_name,
                       'parameters': [p.symbol.name for p in parameters],
                       'input_fields': input_fields,
                       'inputs': self.inputs,
                       'output_fields': output_fields,
                       'other_symbols': other_symbols,
                       'docstring': docstring,
                       'device': 'DEVICE_GPU' if self.is_cuda else 'DEVICE_CPU',
                       'constructor': '',
                       'output_shape': output_shape}
        # TODO dtype -> tf dtype mapping
        compute_method = jinja2.Template(
            """
{%- for f in input_fields -%}
Tensor* {{ f.name }} = (Tensor*) &context->input({{ loop.index - 1 }});
{% endfor -%}
{%- for f in other_symbols -%}
auto _{{ f.name }} = context->input({{ loop.index - 1 + input_fields|length }}).template scalar<{{ f.dtype }}>();
{{ f.dtype }}* {{ f.name }} = ({{ f.dtype }}*) &_{{ f.name }};
{% endfor -%}
{% for f in output_fields -%}
Tensor* {{ f.name }} = nullptr;
OP_REQUIRES_OK(context,
               context->allocate_output({{ loop.index - 1 }}, {{ output_shape }}, &{{ f.name }}));
{% endfor %}
{{ cpp_name }}(
{%- for p in parameters %}
    *{{ p }}{{- ", " if not loop.last -}}
{% endfor %}
);
""").render(render_dict)  # noqa

        super().__init__({**render_dict, 'compute_method': compute_method})

    @property
    def inputs(self):
        return list(itertools.chain(self.input_fields, self.other_symbols))


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
