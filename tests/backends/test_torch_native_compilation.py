# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#

import os
from os.path import dirname, isfile, join

# TODO: from pystencils.backends.cudabackend import generate_cuda
import appdirs
import jinja2
import numpy as np
import torch

import pystencils
import pystencils.autodiff
from pystencils.autodiff.backends._torch_native import (
    create_autograd_function, generate_torch)
# from pystencils.cpu.kernelcreation import create_kernel
from pystencils.backends.cbackend import generate_c
from pystencils.gpucuda.kernelcreation import create_cuda_kernel

PROJECT_ROOT = dirname


def read_file(file):
    with open(file, 'r') as f:
        return f.read()


def test_jit():
    """
    Test JIT compilation from example on git@github.com:pytorch/extension-cpp.git
    `ninja-build` is required
    """

    # os.environ['CUDA_HOME'] = "/usr/local/cuda-10.0"
    cpp_file = join(dirname(__file__), 'lltm_cuda.cpp')
    cuda_file = join(dirname(__file__), 'lltm_cuda_kernel.cu')
    assert isfile(cpp_file)
    assert isfile(cuda_file)

    from torch.utils.cpp_extension import load

    lltm_cuda = load(
        join(dirname(__file__), 'lltm_cuda'), [cpp_file, cuda_file], verbose=True, extra_cuda_cflags=["-ccbin=g++-6"])
    assert lltm_cuda is not None
    print('hallo')


def test_torch_native_compilation():
    x, y = pystencils.fields('x, y: float32[2d]')

    assignments = pystencils.AssignmentCollection({
        y.center(): x.center()**2
    }, {})
    autodiff = pystencils.autodiff.AutoDiffOp(assignments)
    backward_assignments = autodiff.backward_assignments

    print(assignments)
    print(backward_assignments)

    template_string = read_file(join(dirname(__file__),
                                     '../../pystencils/autodiff/backends/torch_native_cuda.tmpl.cpp'))
    template = jinja2.Template(template_string)

    print(template_string)

    forward_kernel = create_cuda_kernel(assignments.all_assignments).body
    backward_kernel = create_cuda_kernel(backward_assignments.all_assignments).body

    forward_code = generate_c(forward_kernel)
    backward_code = generate_c(backward_kernel)

    output = template.render(
        forward_tensors=[f.name for f in autodiff.forward_fields],
        forward_input_tensors=[f.name for f in autodiff.forward_input_fields],
        forward_output_tensors=[f.name for f in autodiff.forward_output_fields],
        backward_tensors=[f.name for f in autodiff.backward_fields + autodiff.forward_input_fields],
        backward_input_tensors=[f.name for f in autodiff.backward_input_fields],
        backward_output_tensors=[f.name for f in autodiff.backward_output_fields],
        forward_kernel=forward_code,
        backward_kernel=backward_code,
        dimensions=range(2),
        kernel_name="square",
        dtype="float"
    )
    print(output)

    template_string = read_file(join(dirname(__file__),
                                     '../../pystencils/autodiff/backends/torch_native_cuda.tmpl.cu'))
    template = jinja2.Template(template_string)

    print(template_string)

    output = template.render(
        forward_tensors=[f.name for f in autodiff.forward_fields],
        forward_input_tensors=[f.name for f in autodiff.forward_input_fields],
        forward_output_tensors=[f.name for f in autodiff.forward_output_fields],
        backward_tensors=[f.name for f in autodiff.backward_fields + autodiff.forward_input_fields],
        backward_input_tensors=[f.name for f in autodiff.backward_input_fields],
        backward_output_tensors=[f.name for f in autodiff.backward_output_fields],
        forward_kernel=forward_code,
        backward_kernel=backward_code,
        kernel_name="square",
        dimensions=range(2)
    )
    print(output)

    template_string = read_file(join(dirname(__file__),
                                     '../../pystencils/autodiff/backends/torch_native_cpu.tmpl.cpp'))
    template = jinja2.Template(template_string)

    print(template_string)

    output = template.render(
        forward_tensors=[f.name for f in autodiff.forward_fields],
        forward_input_tensors=[f.name for f in autodiff.forward_input_fields],
        forward_output_tensors=[f.name for f in autodiff.forward_output_fields],
        backward_tensors=[f.name for f in autodiff.backward_fields + autodiff.forward_input_fields],
        backward_input_tensors=[f.name for f in autodiff.backward_input_fields],
        backward_output_tensors=[f.name for f in autodiff.backward_output_fields],
        forward_kernel=forward_code,
        backward_kernel=backward_code,
        kernel_name="square",
        dtype="float",
        dimensions=range(2)
    )
    print(output)


def test_generate_torch():
    x, y = pystencils.fields('x, y: float32[2d]')

    os.environ['CUDA_HOME'] = "/usr/local/cuda-10.0"
    assignments = pystencils.AssignmentCollection({
        y.center(): x.center()**2
    }, {})
    autodiff = pystencils.autodiff.AutoDiffOp(assignments)

    op_cuda = generate_torch(appdirs.user_cache_dir('pystencils'), autodiff, is_cuda=True, dtype=np.float32)
    assert op_cuda is not None
    op_cpp = generate_torch(appdirs.user_cache_dir('pystencils'), autodiff, is_cuda=False, dtype=np.float32)
    assert op_cpp is not None


def test_execute_torch():
    x, y = pystencils.fields('x, y: float64[32,32]')

    assignments = pystencils.AssignmentCollection({
        y.center(): 5 + x.center()
    }, {})
    autodiff = pystencils.autodiff.AutoDiffOp(assignments)

    x_tensor = pystencils.autodiff.torch_tensor_from_field(x, 1, cuda=False)
    y_tensor = pystencils.autodiff.torch_tensor_from_field(y, 1, cuda=False)

    op_cpp = create_autograd_function(autodiff, {x: x_tensor, y: y_tensor})
    foo = op_cpp.forward(x_tensor)
    print(foo)
    assert op_cpp is not None


def test_execute_torch_gpu():
    x, y = pystencils.fields('x, y: float64[32,32]')

    assignments = pystencils.AssignmentCollection({
        y.center(): 5 + x.center()
    }, {})
    autodiff = pystencils.autodiff.AutoDiffOp(assignments)

    x_tensor = pystencils.autodiff.torch_tensor_from_field(x, 3, cuda=True)
    y_tensor = pystencils.autodiff.torch_tensor_from_field(y, 4, cuda=True)
    assert y_tensor.is_cuda
    assert torch.cuda.is_available()

    op_cuda = create_autograd_function(autodiff, {x: x_tensor, y: y_tensor})
    assert op_cuda is not None
    rtn = op_cuda.forward(y_tensor, x_tensor)
    print(y_tensor)
    print(rtn)


def main():
    # test_jit()
    # test_torch_native_compilation()
    # test_generate_torch()
    test_execute_torch()


main()
