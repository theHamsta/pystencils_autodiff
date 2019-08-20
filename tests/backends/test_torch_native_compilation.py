# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#

import os
import tempfile
from os.path import dirname, isfile, join

import pytest
import sympy

import pystencils
from pystencils_autodiff import create_backward_assignments
from pystencils_autodiff._file_io import _write_file
from pystencils_autodiff.backends.astnodes import TorchModule


def test_torch_jit():
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

    lltm_cuda = load('lltm_cuda', [cpp_file, cuda_file], verbose=True)
    assert lltm_cuda is not None


def test_torch_native_compilation_cpu():
    from torch.utils.cpp_extension import load

    module_name = "Ololol"

    target = 'cpu'

    z, y, x = pystencils.fields("z, y, x: [20,40]")
    a = sympy.Symbol('a')

    forward_assignments = pystencils.AssignmentCollection({
        z[0, 0]: x[0, 0] * sympy.log(a * x[0, 0] * y[0, 0])
    })

    backward_assignments = create_backward_assignments(forward_assignments)

    forward_ast = pystencils.create_kernel(forward_assignments, target)
    forward_ast.function_name = 'forward'
    backward_ast = pystencils.create_kernel(backward_assignments, target)
    backward_ast.function_name = 'backward'
    module = TorchModule(module_name, [forward_ast, backward_ast])
    print(module)

    temp_file = tempfile.NamedTemporaryFile(suffix='.cu' if target == 'gpu' else '.cpp')
    print(temp_file.name)
    _write_file(temp_file.name, str(module))
    torch_extension = load(module_name, [temp_file.name])
    assert torch_extension is not None
    assert 'call_forward' in dir(torch_extension)
    assert 'call_backward' in dir(torch_extension)


def test_torch_native_compilation_gpu():
    from torch.utils.cpp_extension import load

    module_name = "Ololol"

    target = 'gpu'

    z, y, x = pystencils.fields("z, y, x: [20,40]")
    a = sympy.Symbol('a')

    forward_assignments = pystencils.AssignmentCollection({
        z[0, 0]: x[0, 0] * sympy.log(a * x[0, 0] * y[0, 0])
    })

    backward_assignments = create_backward_assignments(forward_assignments)

    forward_ast = pystencils.create_kernel(forward_assignments, target)
    forward_ast.function_name = 'forward'
    backward_ast = pystencils.create_kernel(backward_assignments, target)
    backward_ast.function_name = 'backward'
    module = TorchModule(module_name, [forward_ast, backward_ast])
    print(module)

    temp_file = tempfile.NamedTemporaryFile(suffix='.cu' if target == 'gpu' else '.cpp')
    print(temp_file.name)
    _write_file(temp_file.name, str(module))
    torch_extension = load(module_name, [temp_file.name])
    assert torch_extension is not None
    assert 'call_forward' in dir(torch_extension)
    assert 'call_backward' in dir(torch_extension)


@pytest.mark.skipif(True or 'NO_GPU_EXECUTION' in os.environ, reason='Skip GPU execution tests')
def test_execute_torch_gpu():
    pass
