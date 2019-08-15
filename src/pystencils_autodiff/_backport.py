# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

from pystencils.astnodes import KernelFunction, ResolvedFieldAccess, SympyAssignment


def compatibility_hacks():
    def fields_accessed(self):
        """Set of Field instances: fields which are accessed inside this kernel function"""
        return set(o.field for o in self.atoms(ResolvedFieldAccess))

    def fields_written(self):
        assigments = self.atoms(SympyAssignment)
        return {a.lhs.field for a in assigments if isinstance(a.lhs, ResolvedFieldAccess)}

    KernelFunction.fields_accessed = property(fields_accessed)
    KernelFunction.fields_written = property(fields_written)


compatibility_hacks()
