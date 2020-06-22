import os

import numpy as np
import pytest
import sympy

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
    auto_diff = pystencils_autodiff.AutoDiffOp(assignment_collection,
                                               diff_mode='transposed-forward')
    backward = auto_diff.backward_assignments
    print(backward)
    print('Forward output fields (to check order)')
    print(auto_diff.forward_input_fields)

    print(auto_diff)


@pytest.mark.skipif(True, reason="Temporary skip")
@pytest.mark.xfail(reason="", strict=False)
def test_tfmad_gradient_check():
    tf = pytest.importorskip('tensorflow')

    a, b, out = ps.fields("a, b, out: double[5,6]")
    print(a.shape)

    cont = ps.fd.Diff(a, 0) - ps.fd.Diff(a, 1) - ps.fd.Diff(b, 0) + ps.fd.Diff(
        b, 1)
    discretize = ps.fd.Discretization2ndOrder(dx=1)
    discretization = discretize(cont)

    assignment = ps.Assignment(out.center(), discretization)
    assignment_collection = ps.AssignmentCollection([assignment], [])
    print('Forward')
    print(assignment_collection)

    print('Backward')
    auto_diff = pystencils_autodiff.AutoDiffOp(assignment_collection,
                                               diff_mode='transposed-forward')
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
            [a_tensor, b_tensor], [a.shape, b.shape],
            out_tensor,
            out.shape,
            num_border_pixels=2,
            ndim=2)
        print('error: %s' % gradient_error.max_error)

        assert any(e < 1e-4 for e in gradient_error.values())


def test_tfmad_gradient_check_torch():
    torch = pytest.importorskip('torch')

    a, b, out = ps.fields("a, b, out: float[5,7]")

    cont = 2 * ps.fd.Diff(a, 0) - 1.5 * ps.fd.Diff(a, 1) \
        - ps.fd.Diff(b, 0) + 3 * ps.fd.Diff(b, 1)
    discretize = ps.fd.Discretization2ndOrder(dx=1)
    discretization = discretize(cont) + 1.2*a.center

    assignment = ps.Assignment(out.center(), discretization)
    assignment_collection = ps.AssignmentCollection([assignment], [])
    print('Forward')
    print(assignment_collection)

    print('Backward')
    auto_diff = pystencils_autodiff.AutoDiffOp(assignment_collection,
                                               diff_mode='transposed-forward')
    backward = auto_diff.backward_assignments
    print(backward)
    print('Forward output fields (to check order)')
    print(auto_diff.forward_input_fields)

    a_tensor = torch.zeros(*a.shape, dtype=torch.float64, requires_grad=True)
    b_tensor = torch.zeros(*b.shape, dtype=torch.float64, requires_grad=True)

    function = auto_diff.create_tensorflow_op({
        a: a_tensor,
        b: b_tensor
    }, backend='torch')

    torch.autograd.gradcheck(function.apply, [a_tensor, b_tensor])


@pytest.mark.skip(reason="'valid' seems to be still broken", strict=True)
def test_valid_boundary_handling_tensorflow_native():
    pytest.importorskip('tensorflow')
    import tensorflow as tf

    a, b, out = ps.fields("a, b, out: double[10,11]")
    print(a.shape)

    cont = 2*ps.fd.Diff(a, 0) - 1.5 * ps.fd.Diff(a, 1) - ps.fd.Diff(b, 0) + 3 * ps.fd.Diff(b, 1)
    discretize = ps.fd.Discretization2ndOrder(dx=1)
    discretization = discretize(cont)

    assignment = ps.Assignment(out.center(), discretization + 0.1*b[1, 0])

    assignment_collection = ps.AssignmentCollection([assignment], [])

    print('Forward')
    print(assignment_collection)

    print('Backward')
    auto_diff = pystencils_autodiff.AutoDiffOp(assignment_collection,
                                               boundary_handling='valid')
    backward = auto_diff.backward_assignments
    print(backward)
    print('Forward output fields (to check order)')
    print(auto_diff.forward_input_fields)

    a_tensor = tf.Variable(np.zeros(a.shape, a.dtype.numpy_dtype))
    b_tensor = tf.Variable(np.zeros(a.shape, a.dtype.numpy_dtype))
    # out_tensor = auto_diff.create_tensorflow_op(use_cuda=with_cuda, backend='tensorflow_native')
    # print(out_tensor)

    out_tensor = auto_diff.create_tensorflow_op(use_cuda=False, backend='tensorflow_native')(a=a_tensor, b=b_tensor)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(out_tensor)

        if True:
            gradient_error = compute_gradient_error_without_border(
                [a_tensor, b_tensor], [a.shape, b.shape],
                out_tensor,
                out.shape,
                num_border_pixels=0,
                ndim=2,
                debug=False)
            print('error: %s' % gradient_error.max_error)
            print('avg error: %s' % gradient_error.avg_error)

            assert any(e < 1e-4 for e in gradient_error.values())


@pytest.mark.parametrize('with_offsets', (False, True))
@pytest.mark.parametrize('with_cuda',
                         (False, pytest.param(True, marks=pytest.mark.skipif('CI' in os.environ, reason='PYTORCH does not like pycuda'))))  # noqa
def test_tfmad_gradient_check_torch_native(with_offsets, with_cuda):
    torch = pytest.importorskip('torch')
    import torch

    a, b, out = ps.fields("a, b, out: float64[5,7]")

    if with_offsets:
        cont = 2*ps.fd.Diff(a, 0) - 1.5*ps.fd.Diff(a, 1) - ps.fd.Diff(b, 0) + 3 * ps.fd.Diff(b, 1)
        discretize = ps.fd.Discretization2ndOrder(dx=1)
        discretization = discretize(cont)

        assignment = ps.Assignment(out.center(), discretization + 1.2*a.center())
    else:
        assignment = ps.Assignment(out.center(), 1.2*a.center + 0.1*b.center)

    assignment_collection = ps.AssignmentCollection([assignment], [])
    print('Forward')
    print(assignment_collection)

    print('Backward')
    auto_diff = pystencils_autodiff.AutoDiffOp(assignment_collection,
                                               boundary_handling='zeros',
                                               diff_mode='transposed-forward')
    backward = auto_diff.backward_assignments
    print(backward)
    print('Forward output fields (to check order)')
    print(auto_diff.forward_input_fields)

    a_tensor = torch.zeros(*a.shape, dtype=torch.float64, requires_grad=True).contiguous()
    b_tensor = torch.zeros(*b.shape, dtype=torch.float64, requires_grad=True).contiguous()

    if with_cuda:
        a_tensor = a_tensor.cuda()
        b_tensor = b_tensor.cuda()

    function = auto_diff.create_tensorflow_op(use_cuda=with_cuda, backend='torch_native')

    dict = {
        a: a_tensor,
        b: b_tensor
    }
    torch.autograd.gradcheck(function.apply, tuple(
        [dict[f] for f in auto_diff.forward_input_fields]), atol=1e-4, raise_exception=True)


@pytest.mark.parametrize('with_cuda',
                         (False, pytest.param(True,
                                              marks=pytest.mark.skipif('CI' in os.environ,
                                                                       reason="GPU too old on GITLAB CI"))))
def test_tfmad_gradient_check_two_outputs(with_cuda):
    torch = pytest.importorskip('torch')
    import torch

    a, b, out1, out2, out3 = ps.fields("a, b, out1, out2, out3: float64[21,13]")

    assignment_collection = ps.AssignmentCollection({
        out1.center: a.center + b.center,
        out2.center: a.center - b.center,
        out3.center: sympy.exp(b[-1, 0])
    })
    print('Forward')
    print(assignment_collection)

    print('Backward')
    auto_diff = pystencils_autodiff.AutoDiffOp(assignment_collection,
                                               boundary_handling='zeros',
                                               diff_mode='transposed-forward')
    print(auto_diff.backward_fields)
    backward = auto_diff.backward_assignments
    print(backward)
    print('Forward output fields (to check order)')
    print(auto_diff.forward_input_fields)

    a_tensor = torch.zeros(*a.shape, dtype=torch.float64, requires_grad=True).contiguous()
    b_tensor = torch.zeros(*b.shape, dtype=torch.float64, requires_grad=True).contiguous()
    out1_tensor = torch.zeros(*a.shape, dtype=torch.float64, requires_grad=True).contiguous()
    out2_tensor = torch.zeros(*b.shape, dtype=torch.float64, requires_grad=True).contiguous()
    out3_tensor = torch.zeros(*b.shape, dtype=torch.float64, requires_grad=True).contiguous()

    if with_cuda:
        a_tensor = a_tensor.cuda()
        b_tensor = b_tensor.cuda()
        out1_tensor = out1_tensor.cuda()
        out2_tensor = out2_tensor.cuda()
        out3_tensor = out3_tensor.cuda()

    function = auto_diff.create_tensorflow_op(use_cuda=with_cuda, backend='torch_native')

    dict = {
        a: a_tensor,
        b: b_tensor,
        out1: out1_tensor,
        out2: out2_tensor,
        out3: out3_tensor,
    }
    torch.autograd.gradcheck(function.apply, tuple(
        [dict[f] for f in auto_diff.forward_input_fields]), atol=1e-4, raise_exception=True)


@pytest.mark.parametrize('gradient_check', (False, 'with_gradient_check'))
@pytest.mark.parametrize('with_cuda', (False, pytest.param('with_cuda',
                                                           marks=pytest.mark.skipif('CI' in os.environ,
                                                                                    reason="GPU too old on GITLAB CI"))
                                       ))
@pytest.mark.parametrize('with_offsets', (False, 'with_offsets'))
# @pytest.mark.xfail(reason="", strict=False)
def test_tfmad_gradient_check_tensorflow_native(with_offsets, with_cuda, gradient_check):
    pytest.importorskip('tensorflow')
    import tensorflow as tf

    a, b, out = ps.fields("a, b, out: double[21,13]", layout='fzyx')
    print(a.shape)

    if with_offsets:
        cont = 2*ps.fd.Diff(a, 0) - 1.5 * ps.fd.Diff(a, 1) - ps.fd.Diff(b, 0) + 3 * ps.fd.Diff(b, 1)
        discretize = ps.fd.Discretization2ndOrder(dx=1)
        discretization = discretize(cont)

        assignment = ps.Assignment(out.center(), discretization + 0.1*b[1, 0])

    else:
        assignment = ps.Assignment(out.center(), 1.2*a.center + 0.1*b.center)

    assignment_collection = ps.AssignmentCollection([assignment], [])

    print('Forward')
    print(assignment_collection)

    print('Backward')
    auto_diff = pystencils_autodiff.AutoDiffOp(assignment_collection,
                                               boundary_handling='zeros')
    backward = auto_diff.backward_assignments
    print(backward)
    print('Forward output fields (to check order)')
    print(auto_diff.forward_input_fields)

    # out_tensor = auto_diff.create_tensorflow_op(use_cuda=with_cuda, backend='tensorflow_native')
    # print(out_tensor)

    a_tensor = tf.Variable(np.zeros(a.shape, a.dtype.numpy_dtype))
    b_tensor = tf.Variable(np.zeros(a.shape, a.dtype.numpy_dtype))
    op = auto_diff.create_tensorflow_op(use_cuda=with_cuda, backend='tensorflow_native')

    theoretical, numerical = tf.test.compute_gradient(
        op,
        [a_tensor, b_tensor],
        delta=0.001
    )
    assert np.allclose(theoretical[0], numerical[0])
    assert np.allclose(theoretical[1], numerical[1])


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
    vector_shape = domain_shape + (2, )

    curl_input_for_u = ps.Field.create_fixed_size(field_name='curl_input',
                                                  shape=domain_shape,
                                                  index_dimensions=0)
    u_field = ps.Field.create_fixed_size(field_name='curl',
                                         shape=vector_shape,
                                         index_dimensions=1)

    curl_op = pystencils_autodiff.AutoDiffOp(get_curl(curl_input_for_u,
                                                      u_field),
                                             diff_mode="transposed-forward")

    print('Forward')
    print(curl_op.forward_assignments)

    print('Backward')
    print(curl_op.backward_assignments)
