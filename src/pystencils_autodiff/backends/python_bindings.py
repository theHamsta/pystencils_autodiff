# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import jinja2

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
