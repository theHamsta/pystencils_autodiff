# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import subprocess
import tempfile
from sysconfig import get_paths

import sympy

import pystencils
from pystencils.include import get_pystencils_include_path
from pystencils_autodiff import create_backward_assignments
from pystencils_autodiff._file_io import _write_file
from pystencils_autodiff.backends.astnodes import TensorflowModule


def test_native_tensorflow_compilation_cpu():
    import tensorflow as tf

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
    _write_file(temp_file.name, str(module))
    _write_file('/tmp/foo.cpp', str(module))

    command = ['c++', '-c', '-fPIC', temp_file.name] + compile_flags + link_flags + extra_flags
    print(command)
    subprocess.check_call(command)
    # _pyronn_layers_module = tf.load_op_library(os.path.dirname(__file__)+'/pyronn_layers.so')
    # torch_extension = load(module_name, [temp_file.name])
    # assert torch_extension is not None
    # assert 'call_forward' in dir(torch_extension)
    # assert 'call_backward' in dir(torch_extension)


def test_native_tensorflow_compilation_gpu():
    import tensorflow as tf

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
    forward_ast.function_name = 'forward'
    backward_ast = pystencils.create_kernel(backward_assignments, target)
    backward_ast.function_name = 'backward'
    module = TensorflowModule(module_name, [forward_ast])
    print(module)

    temp_file = tempfile.NamedTemporaryFile(suffix='.cu' if target == 'gpu' else '.cpp')
    print(temp_file.name)
    _write_file(temp_file.name, str(module))
    _write_file('/tmp/foo.cpp', str(module))

    command = ['nvcc', '--expt-relaxed-constexpr', '-DGOOGLE_CUDA=0', '-x', 'cu',
               '-Xcompiler', '-fPIC', '-c', temp_file.name] + compile_flags + link_flags + extra_flags
    print(command)
    subprocess.check_call(command)


def main():
    test_native_tensorflow_compilation_gpu()
