# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import os
import subprocess
import tempfile
from os.path import join
from sysconfig import get_paths

import pytest
import sympy

import pystencils
from pystencils.cpu.cpujit import get_compiler_config
from pystencils.include import get_pystencils_include_path
from pystencils_autodiff import create_backward_assignments
from pystencils_autodiff._file_io import write_file
from pystencils_autodiff.backends.astnodes import TensorflowModule


def test_detect_cpu_vs_cpu():

    for target in ('cpu', 'gpu'):

        z, y, x = pystencils.fields("z, y, x: [20,40]")
        a = sympy.Symbol('a')

        assignments = pystencils.AssignmentCollection({
            z[0, 0]: x[0, 0] * sympy.log(a * x[0, 0] * y[0, 0])
        })
        kernel_ast = pystencils.create_kernel(assignments, target=target)

        module = TensorflowModule('my_module', [kernel_ast])
        print(module)
        assert 'DEVICE_' + target.upper() in str(module)


def test_native_tensorflow_compilation_cpu():
    tf = pytest.importorskip('tensorflow')

    extra_flags = ['-I' + get_paths()['include'], '-I' + get_pystencils_include_path()]

    compile_flags = tf.sysconfig.get_compile_flags()
    link_flags = tf.sysconfig.get_link_flags()

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
    module = TensorflowModule(module_name, [forward_ast, backward_ast])
    print(module)

    temp_file = tempfile.NamedTemporaryFile(suffix='.cu' if target == 'gpu' else '.cpp')
    print(temp_file.name)
    write_file(temp_file.name, str(module))
    write_file('/tmp/foo.cpp', str(module))

    command = ['c++', '-fPIC', temp_file.name, '-O2', '-shared',
               '-o', 'foo.so'] + compile_flags + link_flags + extra_flags
    print(command)
    subprocess.check_call(command)

    lib = tf.load_op_library(join(os.getcwd(), 'foo.so'))
    assert 'call_forward' in dir(lib)
    assert 'call_backward' in dir(lib)


@pytest.mark.skipif("TRAVIS" in os.environ, reason="nvcc compilation currently not working on TRAVIS")
def test_native_tensorflow_compilation_gpu():
    tf = pytest.importorskip('tensorflow')

    extra_flags = ['-I' + get_paths()['include'], '-I' + get_pystencils_include_path()]

    compile_flags = tf.sysconfig.get_compile_flags()
    link_flags = tf.sysconfig.get_link_flags()

    module_name = "Ololol"

    target = 'gpu'

    z, y, x = pystencils.fields("z, y, x: [20,40]")
    a = sympy.Symbol('a')

    forward_assignments = pystencils.AssignmentCollection({
        z[0, 0]: x[0, 0] * sympy.log(a * x[0, 0] * y[0, 0])
    })

    backward_assignments = create_backward_assignments(forward_assignments)

    forward_ast = pystencils.create_kernel(forward_assignments, target)
    forward_ast.function_name = 'forward2'
    backward_ast = pystencils.create_kernel(backward_assignments, target)
    backward_ast.function_name = 'backward2'
    module = TensorflowModule(module_name, [forward_ast, backward_ast])
    print(str(module))

    temp_file = tempfile.NamedTemporaryFile(suffix='.cu' if target == 'gpu' else '.cpp')
    print(temp_file.name)
    write_file(temp_file.name, str(module))
    if 'tensorflow_host_compiler' not in get_compiler_config():
        get_compiler_config()['tensorflow_host_compiler'] = get_compiler_config()['command']

    # on my machine g++-6 and clang-7 are working
    # '-ccbin',
    # 'g++-6',
    command = ['nvcc',
               temp_file.name,
               '-lcudart',
               '--expt-relaxed-constexpr',
               '-ccbin',
               get_compiler_config()['tensorflow_host_compiler'],
               '-lcudart',
               '-std=c++14',
               '-x',
               'cu',
               '-Xcompiler',
               '-fPIC',
               '-c',
               '-o',
               'foo_gpu.o'] + compile_flags + extra_flags

    subprocess.check_call(command)

    # command = ['clang-7', '-shared', temp_file.name, '--cuda-path=/usr/include',  '-std=c++14',
    # '-fPIC', '-lcudart', '-o', 'foo.so'] + compile_flags + link_flags + extra_flags
    command = ['c++', '-fPIC', '-lcudart', 'foo_gpu.o',
               '-shared', '-o', 'foo_gpu.so'] + link_flags

    subprocess.check_call(command)
    lib = tf.load_op_library(join(os.getcwd(), 'foo_gpu.so'))

    assert 'call_forward2' in dir(lib)
    assert 'call_backward2' in dir(lib)