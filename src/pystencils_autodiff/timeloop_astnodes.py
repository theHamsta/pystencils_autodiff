# -*- coding: utf-8 -*-
#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import jinja2
import sympy as sp

from pystencils.astnodes import Node
from pystencils_autodiff.framework_integration.astnodes import FunctionCall, JinjaCppFile


class CustomNode(Node):
    """Own custom base class for all AST nodes."""

    def __init__(self, children):
        self.children = children

    @property
    def args(self):
        """Returns all arguments/children of this node."""
        return self.children

    @property
    def symbols_defined(self):
        return set()

    @property
    def undefined_symbols(self):
        undefined = []

        for c in self.args:
            if hasattr(c, 'undefined_symbols'):
                undefined.extend(c.undefined_symbols)
            else:
                try:
                    c = sp.sympify(c)
                    undefined.extend(c.free_symbols)
                except Exception:
                    pass

        return set(undefined) - self.symbols_defined


class Namespace(CustomNode):
    """Base class for all AST nodes."""

    def __init__(self, name, children):
        self.name = name
        self.children = children

    @property
    def undefined_symbols(self):
        undefined = []

        for c in self.children:
            if hasattr(c, 'undefined_symbols'):
                undefined.extend(c.undefined_symbols)

        return set(undefined)


class Timeloop(CustomNode):

    def __init__(self, loop_symbol, loop_start, loop_end, loop_increment, children: FunctionCall, kernelAstArr=[]):
        self.children = children
        self.loop_symbol = loop_symbol
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.loop_increment = loop_increment
        self.headers = ["<iostream>"]

        self.kernelAstArr = kernelAstArr

    @property
    def args(self):
        """Returns all arguments/children of this node."""
        return (*self.children,
                self.loop_symbol,
                self.loop_start,
                self.loop_end,
                self.loop_increment
                )

    def symbolic_loop_args(self) -> set:
        """ Returns all loop arguments which are not resolved """
        loopArg = [self.loop_start,
                   self.loop_end,
                   self.loop_increment]
        symboliArg = set()

        for arg in loopArg:
            if isinstance(arg, sp.Symbol):
                symboliArg.add(arg)

        return symboliArg

    @property
    def symbols_defined(self):
        return {self.loop_symbol}

    def atoms(self, arg_type):
        """Returns a set of all descendants recursively, which are an instance of the given type."""
        result = set()
        for arg in self.args:
            if isinstance(arg, arg_type):
                result.add(arg)
            if hasattr(arg, 'atoms'):
                result.update(arg.atoms(arg_type))
        return result


class CppNamespace(JinjaCppFile):

    TEMPLATE = jinja2.Template("""
namespace {{ name }} {
{% for c in children %}
   {{ c | indent(3) }}
{% endfor %}
}""")

    def __init__(self, name, children):

        ast_dict = {
            'name': name,
            'children': children
        }

        super().__init__(ast_dict)


class SwapBuffer(CustomNode):
    def __init__(self, first_array, second_array):
        self.first_array = first_array
        self.second_array = second_array
        self.headers = ["<utility>"]

    @property
    def args(self):
        return (self.first_array, self.second_array)
