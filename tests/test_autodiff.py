import sympy as sp

import pystencils as ps
import pystencils_autodiff
from pystencils_autodiff import DiffModes


def test_simple_2d_check_assignment_collection():
    # use simply example
    z, y, x = ps.fields("z, y, x: [2d]")

    forward_assignments = ps.AssignmentCollection([ps.Assignment(
        z[0, 0], x[0, 0]*sp.log(x[0, 0]*y[0, 0]))], [])

    jac = pystencils_autodiff.get_jacobian_of_assignments(
        forward_assignments, [x[0, 0], y[0, 0]])

    assert jac.shape == (len(forward_assignments.bound_symbols),
                         len(forward_assignments.free_symbols))
    print(repr(jac))
    assert repr(jac) == 'Matrix([[log(x_C*y_C) + 1, x_C/y_C]])'

    for diff_mode in DiffModes:
        pystencils_autodiff.create_backward_assignments(
            forward_assignments, diff_mode=diff_mode)
        pystencils_autodiff.create_backward_assignments(
            pystencils_autodiff.create_backward_assignments(forward_assignments), diff_mode=diff_mode)

    result1 = pystencils_autodiff.create_backward_assignments(
        forward_assignments, diff_mode=DiffModes.TRANSPOSED)
    result2 = pystencils_autodiff.create_backward_assignments(
        forward_assignments, diff_mode=DiffModes.TF_MAD)
    assert result1 == result2


def test_simple_2d_check_raw_assignments():
    # use simply example
    z, y, x = ps.fields("z, y, x: [2d]")

    forward_assignments = [ps.Assignment(z[0, 0], x[0, 0]*sp.log(x[0, 0]*y[0, 0]))]

    jac = pystencils_autodiff.get_jacobian_of_assignments(
        forward_assignments, [x[0, 0], y[0, 0]])

    assert jac.shape == (1, 2)
    assert repr(jac) == 'Matrix([[log(x_C*y_C) + 1, x_C/y_C]])'

    for diff_mode in DiffModes:
        pystencils_autodiff.create_backward_assignments(
            forward_assignments, diff_mode=diff_mode)
