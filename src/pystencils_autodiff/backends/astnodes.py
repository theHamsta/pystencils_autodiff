# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import os
import sys
from collections.abc import Iterable
from os.path import dirname, exists, join

import pystencils
from pystencils.astnodes import FieldPointerSymbol, FieldShapeSymbol, FieldStrideSymbol
from pystencils.cpu.cpujit import get_cache_config, get_compiler_config
from pystencils.include import get_pycuda_include_path, get_pystencils_include_path
from pystencils_autodiff._file_io import read_template_from_file, write_file
from pystencils_autodiff.backends.python_bindings import (
    PybindFunctionWrapping, PybindPythonBindings, TensorflowFunctionWrapping,
    TensorflowPythonBindings, TorchPythonBindings)
from pystencils_autodiff.framework_integration.astnodes import (
    DestructuringBindingsForFieldClass, JinjaCppFile, WrapperFunction, generate_kernel_call)
from pystencils_autodiff.tensorflow_jit import _hash


def get_cubic_interpolation_include_paths():
    return [join(dirname(pystencils.gpucuda.__file__), 'CubicInterpolationCUDA', 'code'),
            join(dirname(pystencils.gpucuda.__file__), 'CubicInterpolationCUDA', 'code', 'internal')]


class Header(JinjaCppFile):
    TEMPLATE = read_template_from_file(join(dirname(__file__), 'module.tmpl.hpp'))

    def __init__(self, exported_functions, module_name):
        ast_dict = {
            'declarations': exported_functions,
            'module_name': module_name,
        }

        super().__init__(ast_dict)

    def __str__(self):
        self.printer._signatureOnly = True
        rtn = JinjaCppFile.__str__(self)
        self.printer._signatureOnly = False
        return rtn

    @property
    def backend(self):
        return 'c'


class TorchTensorDestructuring(DestructuringBindingsForFieldClass):
    CLASS_TO_MEMBER_DICT = {
        FieldPointerSymbol: "data_ptr<{dtype}>()",
        FieldShapeSymbol: "size({dim})",
        FieldStrideSymbol: "strides()[{dim}]"
    }

    CLASS_NAME_TEMPLATE = "at::Tensor"

    headers = ["<ATen/ATen.h>"]


class TensorflowTensorDestructuring(DestructuringBindingsForFieldClass):
    CLASS_TO_MEMBER_DICT = {
        FieldPointerSymbol: "flat<{dtype}>().data()",
        FieldShapeSymbol: "dim_size({dim})",
        FieldStrideSymbol: "dim_size({dim})"
    }

    CLASS_NAME_TEMPLATE = "tensorflow::Tensor"

    headers = ["<tensorflow/core/framework/tensor.h>",
               "<tensorflow/core/framework/types.h>",
               ]


class PybindArrayDestructuring(DestructuringBindingsForFieldClass):
    CLASS_TO_MEMBER_DICT = {
        FieldPointerSymbol: "mutable_data()",
        FieldShapeSymbol: "shape({dim})",
        FieldStrideSymbol: "strides({dim}) / sizeof({dtype})"
    }

    CLASS_NAME_TEMPLATE = "pybind11::array_t<{dtype}>"
    ARGS_AS_REFERENCE = False

    headers = ["<pybind11/numpy.h>"]


class TorchModule(JinjaCppFile):
    TEMPLATE = read_template_from_file(join(dirname(__file__), 'module.tmpl.cpp'))
    DESTRUCTURING_CLASS = TorchTensorDestructuring
    PYTHON_BINDINGS_CLASS = TorchPythonBindings
    PYTHON_FUNCTION_WRAPPING_CLASS = PybindFunctionWrapping

    @property
    def backend(self):
        return 'gpucuda' if self.is_cuda else 'c'

    def __init__(self,
                 module_name,
                 kernel_asts,
                 with_python_bindings=True,
                 wrap_wrapper_functions=False,
                 class_definitions=[]):
        """Create a C++ module with forward and optional backward_kernels

        :param forward_kernel_ast: one or more kernel ASTs (can have any C dialect)
        :param backward_kernel_ast:
        """
        from pystencils_autodiff.framework_integration.astnodes import CustomFunctionCall

        if not isinstance(kernel_asts, Iterable):
            kernel_asts = [kernel_asts]
        wrapper_functions = [self.generate_wrapper_function(k)
                             if not isinstance(k, WrapperFunction)
                             else k for k in kernel_asts]
        kernel_asts = [k for k in kernel_asts if not isinstance(k, (WrapperFunction, CustomFunctionCall))]
        self.module_name = module_name
        self.compiled_file = None

        ast_dict = {
            'kernels': kernel_asts,
            'kernel_wrappers': wrapper_functions,
            'module_name': module_name,
            'python_bindings': self.PYTHON_BINDINGS_CLASS(module_name,
                                                          [self.PYTHON_FUNCTION_WRAPPING_CLASS(a)
                                                              for a in wrapper_functions] + class_definitions)
            if with_python_bindings else ''
        }

        super().__init__(ast_dict)

    @property
    def kernel_wrappers(self):
        return self.ast_dict['kernel_wrappers']

    @classmethod
    def generate_wrapper_function(cls, kernel_ast):
        return WrapperFunction(cls.DESTRUCTURING_CLASS(generate_kernel_call(kernel_ast)),
                               function_name='call_' + kernel_ast.function_name)

    def compile(self,
                extra_source_files=[],
                extra_cuda_flags=[],
                with_cuda=None,
                build_dir=None,
                compile_module_name=None):
        from torch.utils.cpp_extension import load
        file_extension = '.cu' if self.is_cuda else '.cpp'
        source_code = str(self)
        hash = _hash(source_code.encode()).hexdigest()
        if not build_dir:
            build_dir = join(get_cache_config()['object_cache'], self.module_name)
        os.makedirs(build_dir, exist_ok=True)
        file_name = join(build_dir, f'{hash}{file_extension}')

        self.compiled_file = join(build_dir, compile_module_name or file_name).replace('.cpp', '') + '.so'

        if not exists(file_name):
            write_file(file_name, source_code)

        # Torch regards CXX
        os.environ['CXX'] = get_compiler_config()['command']

        torch_extension = load(compile_module_name or hash,
                               [file_name] + extra_source_files,
                               with_cuda=self.is_cuda or with_cuda,
                               extra_cflags=['--std=c++14', get_compiler_config()
                                             ['flags'].replace('--std=c++11', '')],
                               extra_cuda_cflags=['-std=c++14', '-ccbin',
                                                  get_compiler_config()['command']] + extra_cuda_flags,
                               build_directory=build_dir,
                               extra_include_paths=[get_pycuda_include_path(),
                                                    get_pystencils_include_path(),
                                                    *get_cubic_interpolation_include_paths()])
        return torch_extension

    @property
    def header(self):
        return Header(self.ast_dict.kernel_wrappers, self.module_name)


class TensorflowModule(TorchModule):
    DESTRUCTURING_CLASS = TensorflowTensorDestructuring
    PYTHON_BINDINGS_CLASS = TensorflowPythonBindings
    PYTHON_FUNCTION_WRAPPING_CLASS = TensorflowFunctionWrapping

    def __init__(self, module_name, kernel_asts):
        """Create a C++ module with forward and optional backward_kernels

        Args:
            module_name (str): Module name
            kernel_asts (pystencils.kernel_wrappers.KernelWrapper):
                        ASTs as generated by `:func:pystencils.create_kernel`
        """

        self.compiled_file = None
        super().__init__(module_name, kernel_asts)

    def compile(self):
        from pystencils_autodiff.tensorflow_jit import compile_sources_and_load
        if self.is_cuda:
            self.compiled_file = compile_sources_and_load([], cuda_sources=[str(self)], compile_only=True)
        else:
            self.compiled_file = compile_sources_and_load([str(self)], compile_only=True)
        import tensorflow as tf
        return tf.load_op_library(self.compiled_file)


class PybindModule(TorchModule):
    DESTRUCTURING_CLASS = PybindArrayDestructuring
    PYTHON_BINDINGS_CLASS = PybindPythonBindings

    CPP_IMPORT_PREFIX = """/*
<%
setup_pybind11(cfg)
%>
*/
"""

    def compile(self):
        try:
            import cppimport
        except ImportError:
            try:
                from torch.utils.cpp_extension import load
            except Exception:
                assert False, 'cppimport or torch ist required for compiling pybind11 modules'

        assert not self.is_cuda

        source_code = self.CPP_IMPORT_PREFIX + str(self)
        hash_str = _hash(source_code.encode()).hexdigest()
        source_code_with_hash = source_code.replace(
            f'PYBIND11_MODULE({self.module_name}',
            f'PYBIND11_MODULE(cppimport_{hash_str}')

        cache_dir = join(get_cache_config()['object_cache'])
        file_name = join(cache_dir, f'cppimport_{hash_str}.cpp')

        os.makedirs(cache_dir, exist_ok=True)
        if not exists(file_name):
            write_file(file_name, source_code_with_hash)
        # TODO: propagate extra headers
        if cache_dir not in sys.path:
            sys.path.append(cache_dir)

        # Torch regards CXX
        os.environ['CXX'] = get_compiler_config()['command']

        try:
            torch_extension = cppimport.imp(f'cppimport_{hash_str}')
        except Exception as e:
            print(e)
            torch_extension = load(hash,
                                   [file_name],
                                   with_cuda=self.is_cuda,
                                   extra_cflags=['--std=c++14',
                                                 get_compiler_config()['flags'].replace('--std=c++11', '')],
                                   extra_cuda_cflags=['-std=c++14', '-ccbin', get_compiler_config()['command']],
                                   build_directory=cache_dir,
                                   extra_include_paths=[get_pycuda_include_path(),
                                                        get_pystencils_include_path()])
        return torch_extension
