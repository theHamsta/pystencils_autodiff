#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from pystencils.data_types import Type


class TemplateType(Type):

    def __init__(self, name):
        self._name = name

    def _sympystr(self, *args, **kwargs):
        return str(self._name)


class CustomCppType(Type):

    def __init__(self, name):
        self._name = name

    def _sympystr(self, *args, **kwargs):
        return str(self._name)
