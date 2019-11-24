import itertools

import sympy as sp
from sympy.matrices.dense import matrix_multiply_elementwise

import pystencils
from pystencils import Field, x_vector
from pystencils.astnodes import ConditionalFieldAccess
from pystencils.simp import sympy_cse


def add_fixed_constant_boundary_handling(assignments, with_cse=True):

    field_accesses = set().union(itertools.chain.from_iterable(
        [a.atoms(Field.Access) for a in assignments]
    ))

    if all(all(o == 0 for o in a.offsets) for a in field_accesses):
        return assignments
    common_shape = next(iter(field_accesses)).field.spatial_shape
    ndim = len(common_shape)

    def is_out_of_bound(access, shape):
        return sp.Or(*[sp.Or(a < 0, a >= s) for a, s in zip(access, shape)])

    safe_assignments = [pystencils.Assignment(
        assignment.lhs, assignment.rhs.subs({
            a: ConditionalFieldAccess(a, is_out_of_bound(sp.Matrix(a.offsets) + x_vector(ndim), common_shape))
            for a in assignment.rhs.atoms(Field.Access) if not a.is_absolute_access
        })) for assignment in assignments.all_assignments]

    if with_cse:
        safe_assignments = sympy_cse(pystencils.AssignmentCollection(safe_assignments))
        return safe_assignments
    else:
        return pystencils.AssignmentCollection(safe_assignments)


def get_random_sampling(random_numbers, aabb_min, aabb_max):
    random_numbers = sp.Matrix(random_numbers)
    aabb_min = sp.Matrix(aabb_min)
    aabb_max = sp.Matrix(aabb_max)
    return matrix_multiply_elementwise(random_numbers, (aabb_max - aabb_min)) + aabb_min
