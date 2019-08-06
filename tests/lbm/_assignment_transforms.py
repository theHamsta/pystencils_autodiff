from typing import List
import pystencils as ps


def shift_assignments(assignments: List[ps.Assignment], shift_vector, shift_write_assignments=True, shift_read_assignments=True):
    if hasattr(assignments, "main_assignments"):
        assignments = assignments.main_assignments

    shifted_assignments = assignments
    for i, a in enumerate(shifted_assignments):

        for symbol in [*a.free_symbols, a.rhs]:
            if isinstance(symbol, ps.Field.Access):
                shifted_access = symbol.get_shifted(
                    *shift_vector)
            shifted_assignments[i] = shifted_assignments[i].subs(
                symbol, shifted_access)
    return shifted_assignments


def transform_assignments_rhs(assignments, rhs_transform_function):
    new_assignments = []
    for _, a in enumerate(assignments):
        new_rhs = rhs_transform_function(a)

        new_assignments.append(
            ps.Assignment(a.lhs, new_rhs))

    return new_assignments


def transform_assignments(assignments, transform_function):
    new_assignments = []
    for _, a in enumerate(assignments):
        new_assignment = transform_function(a)

        if isinstance(new_assignment, tuple):
            lhs, rhs = new_assignment
            new_assignment = ps.Assignment(lhs, rhs)

        if new_assignment is not None:
            new_assignments.append(new_assignment)

    return new_assignments


def replace_symbols_in_assignments(assignments: List[ps.Assignment], symbol, replace_symbol):
    if hasattr(assignments, "main_assignments"):
        assignments = assignments.main_assignments

    shifted_assignments = assignments
    for i, a in enumerate(shifted_assignments):
        shifted_assignments[i] = a.subs(symbol, replace_symbol)
    return shifted_assignments
