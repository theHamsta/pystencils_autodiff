import argparse
import os

import numpy as np
import pytest
import sympy as sp
import tensorflow as tf
import torch

import pystencils as ps
import pystencils_autodiff
from test_utils.gradient_check_tensorflow import compute_gradient_error_without_border


def test_tfmad_stencil():

    f, out = ps.fields("f, out: double[2D]")

    cont = ps.fd.Diff(f, 0) - ps.fd.Diff(f, 1)
    discretize = ps.fd.Discretization2ndOrder(dx=1)
    discretization = discretize(cont)

    assignment = ps.Assignment(out.center(), discretization)
    assignment_collection = ps.AssignmentCollection([assignment], [])
    print('Forward')
    print(assignment_collection)

    print('Backward')
    backward = pystencils_autodiff.create_backward_assignments(
        assignment_collection, diff_mode='transposed-forward')
    print(backward)


def test_tfmad_two_stencils():

    a, b, out = ps.fields("a, b, out: double[2D]")

    cont = ps.fd.Diff(a, 0) - ps.fd.Diff(a, 1) - \
        ps.fd.Diff(b, 0) + ps.fd.Diff(b, 1)
    discretize = ps.fd.Discretization2ndOrder(dx=1)
    discretization = discretize(cont)

    assignment = ps.Assignment(out.center(), discretization)
    assignment_collection = ps.AssignmentCollection([assignment], [])
    print('Forward')
    print(assignment_collection)

    print('Backward')
    auto_diff = pystencils_autodiff.AutoDiffOp(
        assignment_collection, diff_mode='transposed-forward')
    backward = auto_diff.backward_assignments
    print(backward)
    print('Forward output fields (to check order)')
    print(auto_diff.forward_input_fields)

    print(auto_diff)


@pytest.mark.skipif("NO_TENSORFLOW_TEST" in os.environ, reason="Requires Tensorflow")
def test_tfmad_gradient_check():
    a, b, out = ps.fields("a, b, out: double[21,13]")
    print(a.shape)

    cont = ps.fd.Diff(a, 0) - ps.fd.Diff(a, 1) - ps.fd.Diff(b, 0) + ps.fd.Diff(b, 1)
    discretize = ps.fd.Discretization2ndOrder(dx=1)
    discretization = discretize(cont)

    assignment = ps.Assignment(out.center(), discretization)
    assignment_collection = ps.AssignmentCollection([assignment], [])
    print('Forward')
    print(assignment_collection)

    print('Backward')
    auto_diff = pystencils_autodiff.AutoDiffOp(
        assignment_collection, diff_mode='transposed-forward')
    backward = auto_diff.backward_assignments
    print(backward)
    print('Forward output fields (to check order)')
    print(auto_diff.forward_input_fields)

    a_tensor = tf.Variable(np.zeros(a.shape, a.dtype.numpy_dtype))
    b_tensor = tf.Variable(np.zeros(a.shape, a.dtype.numpy_dtype))
    out_tensor = auto_diff.create_tensorflow_op({a: a_tensor, b: b_tensor})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        gradient_error = compute_gradient_error_without_border(
            [a_tensor, b_tensor], [a.shape, b.shape], out_tensor, out.shape, num_border_pixels=2, ndim=2)
        print('error: %s' % gradient_error.max_error)

        assert any(e < 1e-4 for e in gradient_error.values())


def check_tfmad_vector_input_data(args):
    dtype = args.dtype
    domain_shape = args.domain_shape
    ndim = len(domain_shape)

    # create arrays
    c_arr = np.zeros(domain_shape)
    v_arr = np.zeros(domain_shape + (ndim,))

    # create fields
    c, v, c_next = ps.fields("c, v(2), c_next: % s[ % i, % i]" % ("float" if dtype == np.float32 else "double",
                                                                  domain_shape[0],
                                                                  domain_shape[1]),
                             c=c_arr,
                             v=v_arr,
                             c_next=c_arr)

    # write down advection diffusion pde
    # the equation is represented by a single term and an implicit "=0" is assumed.
    adv_diff_pde = ps.fd.transient(
        c) - ps.fd.diffusion(c, sp.Symbol("D")) + ps.fd.advection(c, v)

    discretize = ps.fd.Discretization2ndOrder(args.dx, args.dt)
    discretization = discretize(adv_diff_pde)
    discretization = discretization.subs(
        sp.Symbol("D"), args.diffusion_coefficient)
    forward_assignments = ps.AssignmentCollection(
        [ps.Assignment(c_next.center(), discretization)], [])

    autodiff = pystencils_autodiff.AutoDiffOp(
        forward_assignments, diff_mode='transposed-forward')  # , constant_fields=[v]

    print('Forward assignments:')
    print(autodiff.forward_assignments)
    print('Backward assignments:')
    print(autodiff.backward_assignments)


def test_tfmad_vector_input_data():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--domain_shape', default=(100, 30), nargs=2, type=int, help="")
    parser.add_argument(
        '--dx', default=1, type=float, help="")
    parser.add_argument(
        '--dt', default=0.01, type=float, help="")
    parser.add_argument(
        '--diffusion_coefficient', default=1, type=float, help="")
    parser.add_argument(
        '--num_total_time_steps', default=100, type=int)
    parser.add_argument(
        '--num_time_steps_for_op', default=1, type=int)
    parser.add_argument(
        '--learning_rate', default=1e-2, type=float)
    parser.add_argument(
        '--dtype', default=np.float64, type=np.dtype)
    parser.add_argument(
        '--num_optimization_steps', default=2000, type=int)
    parser.add_argument('vargs', nargs='*')

    args = parser.parse_args()
    check_tfmad_vector_input_data(args)


@pytest.mark.skipif("NO_TORCH_TEST" in os.environ, reason="Requires PyTorch")
def test_tfmad_gradient_check_torch():
    a, b, out = ps.fields("a, b, out: float[21,13]")

    cont = ps.fd.Diff(a, 0) - ps.fd.Diff(a, 1) - \
        ps.fd.Diff(b, 0) + ps.fd.Diff(b, 1)
    discretize = ps.fd.Discretization2ndOrder(dx=1)
    discretization = discretize(cont)

    assignment = ps.Assignment(out.center(), discretization)
    assignment_collection = ps.AssignmentCollection([assignment], [])
    print('Forward')
    print(assignment_collection)

    print('Backward')
    auto_diff = pystencils_autodiff.AutoDiffOp(
        assignment_collection, diff_mode='transposed-forward')
    backward = auto_diff.backward_assignments
    print(backward)
    print('Forward output fields (to check order)')
    print(auto_diff.forward_input_fields)

    a_tensor = torch.zeros(
        *a.shape, dtype=torch.float64, requires_grad=True)
    b_tensor = torch.zeros(
        *b.shape, dtype=torch.float64, requires_grad=True)
    function = auto_diff.create_tensorflow_op(
        {a: a_tensor, b: b_tensor}, backend='torch')

    torch.autograd.gradcheck(function.apply, [a_tensor, b_tensor])


def get_curl(input_field: ps.Field, curl_field: ps.Field):
    """Return a ps.AssignmentCollection describing the calculation of
    the curl given a 2d or 3d vector field [z,y,x](f) or [y,x](f)

    Note that the curl of a 2d vector field is defined in ℝ3!
    Only the non-zero z-component is returned

    Arguments:
        field {ps.Field} -- A field with index_dimensions <= 1
            Scalar fields are interpreted as a z-component

    Raises:
        NotImplementedError -- [description]
        NotImplementedError -- Only support 2d or 3d vector fields or scalar fields are supported

    Returns:
        ps.AssignmentCollection -- AssignmentCollection describing the calculation of the curl
    """
    assert input_field.index_dimensions <= 1, "Must be a vector or a scalar field"
    assert curl_field.index_dimensions == 1, "Must be a vector field"
    discretize = ps.fd.Discretization2ndOrder(dx=1)

    if input_field.index_dimensions == 0:
        dy = ps.fd.Diff(input_field, 0)
        dx = ps.fd.Diff(input_field, 1)
        f_x = ps.Assignment(curl_field.center(0), discretize(dy))
        f_y = ps.Assignment(curl_field.center(1), discretize(dx))
        return ps.AssignmentCollection([f_x, f_y], [])

    else:

        if input_field.index_shape[0] == 2:
            raise NotImplementedError()

        elif input_field.index_shape[0] == 3:
            raise NotImplementedError()
        else:
            raise NotImplementedError()


def test_tfmad_two_outputs():

    domain_shape = (20, 30)
    vector_shape = domain_shape + (2,)

    curl_input_for_u = ps.Field.create_fixed_size(
        field_name='curl_input', shape=domain_shape, index_dimensions=0)
    u_field = ps.Field.create_fixed_size(
        field_name='curl', shape=vector_shape, index_dimensions=1)

    curl_op = pystencils_autodiff.AutoDiffOp(get_curl(
        curl_input_for_u, u_field), diff_mode="transposed-forward")

    print('Forward')
    print(curl_op.forward_assignments)

    print('Backward')
    print(curl_op.backward_assignments)


def main():
    # test_tfmad_stencil()
    # test_tfmad_two_stencils()
    # test_tfmad_gradient_check_torch()
    test_tfmad_gradient_check()
    # test_tfmad_vector_input_data()
    # test_tfmad_two_outputs()


if __name__ == '__main__':
    main()
