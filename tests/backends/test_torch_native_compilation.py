# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#

import os
import subprocess
from os.path import dirname, isfile, join

import numpy as np
import pytest
import sympy

import pystencils
from pystencils_autodiff import create_backward_assignments
from pystencils_autodiff._file_io import write_cached_content
from pystencils_autodiff.backends.astnodes import PybindModule, TorchModule

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(subprocess.call(['ninja', '--version']) != 0,
                                reason='torch compilation requires ninja')


PROJECT_ROOT = dirname


@pytest.mark.skipif("TRAVIS" in os.environ, reason="nvcc compilation currently not working on TRAVIS")
@pytest.mark.skipif("GITLAB_CI" in os.environ, reason="Gitlab GPUs with CC 3.0 to old!")
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

    temp_file = write_cached_content(str(module), '.cpp')
    torch_extension = load(module_name, [temp_file])
    assert torch_extension is not None
    assert 'call_forward' in dir(torch_extension)
    assert 'call_backward' in dir(torch_extension)

    torch_extension = module.compile()
    assert torch_extension is not None
    assert 'call_forward' in dir(torch_extension)
    assert 'call_backward' in dir(torch_extension)


@pytest.mark.parametrize('with_python_bindings', ('with_python_bindings', False))
@pytest.mark.skipif("GITLAB_CI" in os.environ, reason="Gitlab GPUs with CC 3.0 to old!")
def test_pybind11_compilation_cpu(with_python_bindings):

    pytest.importorskip('pybind11')
    pytest.importorskip('cppimport')

    module_name = "Olololsada"

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
    module = PybindModule(module_name, [forward_ast, backward_ast], with_python_bindings=with_python_bindings)
    print(module)

    if with_python_bindings:
        pybind_extension = module.compile()
        assert pybind_extension is not None
        assert 'call_forward' in dir(pybind_extension)
        assert 'call_backward' in dir(pybind_extension)


@pytest.mark.skipif("TRAVIS" in os.environ, reason="nvcc compilation currently not working on TRAVIS")
@pytest.mark.skipif("GITLAB_CI" in os.environ, reason="Gitlab GPUs with CC 3.0 to old!")
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

    temp_file = write_cached_content(str(module), suffix='.cu')
    torch_extension = load(module_name, [temp_file])
    assert torch_extension is not None
    assert 'call_forward' in dir(torch_extension)
    assert 'call_backward' in dir(torch_extension)

    torch_extension = module.compile()
    assert torch_extension is not None
    assert 'call_forward' in dir(torch_extension)
    assert 'call_backward' in dir(torch_extension)


@pytest.mark.parametrize('target', (
    pytest.param('gpu', marks=pytest.mark.skipif('CI' in os.environ, reason="GPU too old on GITLAB CI")),
    'cpu'))
def test_execute_torch(target):
    import pycuda.autoinit
    module_name = "Ololol" + target

    z, y, x = pystencils.fields("z, y, x: [20,40]")
    a = sympy.Symbol('a')

    forward_assignments = pystencils.AssignmentCollection({
        z[0, 0]: x[0, 0] * sympy.log(a * x[0, 0] * y[0, 0])
    })

    # backward_assignments = create_backward_assignments(forward_assignments)

    if target == 'cpu':
        x = np.random.rand(20, 40)
        y = np.random.rand(20, 40)
        z = np.zeros((20, 40))
    else:
        gpuarray = pytest.importorskip('pycuda.gpuarray')
        x = gpuarray.to_gpu(np.random.rand(20, 40))
        y = gpuarray.to_gpu(np.random.rand(20, 40))
        z = gpuarray.zeros((20, 40), np.float64)

    kernel = pystencils.create_kernel(forward_assignments, target=target)
    kernel.function_name = 'forward'

    torch_module = TorchModule(module_name, [kernel]).compile()
    pystencils_module = kernel.compile()

    pystencils_module(x=x, y=y, z=z, a=5.)
    if target == 'gpu':
        x = x.get()
        y = y.get()
        z = z.get()

    z_pystencils = np.copy(z)
    import torch
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    z = torch.Tensor(z)
    if target == 'gpu':
        x = x.double().cuda()
        y = y.double().cuda()
        z = z.double().cuda()
    else:
        x = x.double()
        y = y.double()
        z = z.double()

    torch_module.call_forward(x=x, y=y, z=z, a=5.)
    if target == 'gpu':
        z = z.cpu()

    z_torch = np.copy(z)

    assert np.allclose(z_torch[1:-1, 1:-1], z_pystencils[1:-1, 1:-1], atol=1e-6)


def test_reproducability():
    from sympy.core.cache import clear_cache

    output_0 = None
    for i in range(10):
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
        new_output = str(TorchModule(module_name, [forward_ast, backward_ast]))
        TorchModule(module_name, [forward_ast, backward_ast]).compile()

        clear_cache()

        if not output_0:
            output_0 = new_output

        assert output_0 == new_output


def test_fields_from_torch_tensor():
    torch = pytest.importorskip('torch')
    import torch
    a, b = torch.zeros((20, 10)), torch.zeros((6, 7))
    x, y = pystencils.fields(x=a, y=b)
    print(x)
    print(y)
    c = torch.zeros((20, 10)).cuda()
    z = pystencils.fields(z=c)
    print(z)
