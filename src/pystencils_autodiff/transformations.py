# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import itertools

import sympy as sp

import pystencils
from pystencils import Field, x_vector
from pystencils.astnodes import ConditionalFieldAccess
from pystencils.simp import sympy_cse


def add_fixed_constant_boundary_handling(assignments, with_cse=True):

    common_shape = next(iter(set().union(itertools.chain.from_iterable(
        [a.atoms(Field.Access) for a in assignments]
    )))).field.spatial_shape
    ndim = len(common_shape)

    def is_out_of_bound(access, shape):
        return sp.Or(*[sp.Or(a < 0, a >= s) for a, s in zip(access, shape)])

    safe_assignments = [pystencils.Assignment(
        assignment.lhs, assignment.rhs.subs({
            a: ConditionalFieldAccess(a, is_out_of_bound(sp.Matrix(a.offsets) + x_vector(ndim), common_shape))
            for a in assignment.rhs.atoms(Field.Access) if not a.is_absolute_access
        })) for assignment in assignments.all_assignments]

    subs = [{a: ConditionalFieldAccess(a, is_out_of_bound(
        sp.Matrix(a.offsets) + x_vector(ndim), common_shape))
        for a in assignment.rhs.atoms(Field.Access) if not a.is_absolute_access
    } for assignment in assignments.all_assignments]
    print(subs)

    if with_cse:
        safe_assignments = sympy_cse(pystencils.AssignmentCollection(safe_assignments))
        return safe_assignments
    else:
        return pystencils.AssignmentCollection(safe_assignments)
