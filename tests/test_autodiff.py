import pytest
import sympy as sp

import pystencils as ps
import pystencils.autodiff

<< << << < HEAD
#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Stephan Seitz"
__copyright__ = "Stephan Seitz"
__license__ = "GPL-v3"


def test_simple_2d_check_assignment_collection():
    # use simply example
    z, x, y = ps.fields("z, y, x: [2d]")

    forward_assignments = ps.AssignmentCollection([ps.Assignment(
        z[0, 0], x[0, 0]*sp.log(x[0, 0]*y[0, 0]))], [])

    jac = pystencils.autodiff.get_jacobian_of_assignments(
        forward_assignments, [x[0, 0], y[0, 0]])

    assert jac.shape == (len(forward_assignments.bound_symbols),
                         len(forward_assignments.free_symbols))
    print(repr(jac))
    assert repr(jac) == 'Matrix([[log(x_C*y_C) + 1, y_C/x_C]])'

    pystencils.autodiff.create_backward_assignments(
        forward_assignments)
    pystencils.autodiff.create_backward_assignments(
        pystencils.autodiff.create_backward_assignments(forward_assignments))


def test_simple_2d_check_raw_assignments():
    # use simply example
    z, x, y = ps.fields("z, y, x: [2d]")

    forward_assignments = \
        [ps.Assignment(z[0, 0], x[0, 0]*sp.log(x[0, 0]*y[0, 0]))]

    jac = pystencils.autodiff.get_jacobian_of_assignments(
        forward_assignments, [x[0, 0], y[0, 0]])

    assert jac.shape == (1, 2)
    assert repr(jac) == 'Matrix([[log(x_C*y_C) + 1, y_C/x_C]])'

    pystencils.autodiff.create_backward_assignments(
        forward_assignments)


def main():
    test_simple_2d_check_assignment_collection()
    test_simple_2d_check_raw_assignments()


if __name__ == '__main__':
    main()
