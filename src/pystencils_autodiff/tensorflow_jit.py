# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import hashlib
import os
import subprocess
import sysconfig
from os.path import exists, join

import p_tqdm

import pystencils
import pystencils.gpucuda
from pystencils.cpu.cpujit import get_cache_config, get_compiler_config
from pystencils.include import get_pycuda_include_path, get_pystencils_include_path
from pystencils_autodiff._file_io import read_file, write_file

_hash = hashlib.md5

if 'NVCC_BINARY' in os.environ:
    NVCC_BINARY = os.environ['NVCC_BINARY']
else:
    NVCC_BINARY = 'nvcc'

# TODO: msvc
if get_compiler_config()['os'] != 'windows':
    _shared_object_flag = '-shared'
    _output_flag = '-o'
    _include_flags = ['-I' + sysconfig.get_paths()['include'],
                      '-I' + get_pystencils_include_path(),
                      '-I' + get_pycuda_include_path()]
    _do_not_link_flag = "-c"
    _position_independent_flag = "-fPIC"
    _compile_env = os.environ.copy()
    _object_file_extension = '.o'
    _link_cudart = '-lcudart'
    _openmp_flag = '-fopenmp'
else:
    _do_not_link_flag = '-c'
    _output_flag = '-o'
    _shared_object_flag = '/DLL'
    _include_flags = ['-I' + sysconfig.get_paths()['include'],
                      '-I' + get_pystencils_include_path(),
                      '-I' + get_pycuda_include_path()]
    _position_independent_flag = "/DTHIS_FLAG_DOES_NOTHING"
    get_compiler_config()['command'] = 'cl.exe'
    config_env = get_compiler_config()['env'] if 'env' in get_compiler_config() else {}
    _compile_env = os.environ.copy()
    _compile_env.update(config_env)
    _object_file_extension = '.obj'
    _link_cudart = '/link cudart'  # ???
    _openmp_flag = '/openmp'  # ???


def link(object_files,
         destination_file=None,
         overwrite_destination_file=True,
         additional_link_flags=[],
         link_cudart=False):
    """Compiles given :param:`source_file` to a Tensorflow shared Library.

    .. warning::

        This functions performs no caching. Only call when your source_files changed!

    :sources_file: Files to compile
    :destination_file: Optional destination path and filename for shared object.
    :returns: Object containing all Tensorflow Ops in that shared library.

    """
    import tensorflow as tf
    command_prefix = [get_compiler_config()['command'],
                      _position_independent_flag,
                      *object_files,
                      *tf.sysconfig.get_link_flags(),
                      *_include_flags,
                      *additional_link_flags,
                      _shared_object_flag,
                      _output_flag]
    if not destination_file:
        destination_file = join(get_cache_config()['object_cache'],
                                f"{_hash('.'.join(sorted(object_files + command_prefix)).encode()).hexdigest()}.so")

    if not exists(destination_file) or overwrite_destination_file:
        command = command_prefix + [destination_file]
        if link_cudart:
            command.append(_link_cudart)
        subprocess.check_call(command, env=_compile_env)

    return destination_file


def link_and_load(object_files,
                  destination_file=None,
                  overwrite_destination_file=True,
                  additional_link_flags=[],
                  link_cudart=False):
    import tensorflow as tf

    destination_file = link(object_files,
                            destination_file,
                            overwrite_destination_file,
                            additional_link_flags,
                            link_cudart)
    lib = tf.load_op_library(destination_file)
    return lib


def try_get_cuda_arch_flag():
    if 'PYSTENCILS_TENSORFLOW_NVCC_ARCH' in os.environ:
        return "-arch=sm_" + os.environ['PYSTENCILS_TENSORFLOW_NVCC_ARCH']
    try:
        from pycuda.driver import Context
        arch = "sm_%d%d" % Context.get_device().compute_capability()
    except Exception:
        arch = None
    if arch:
        return "-arch=" + arch
    else:
        return None


_cuda_arch_flag = try_get_cuda_arch_flag()

_nvcc_flags = ["-w", "-std=c++14", "-Wno-deprecated-gpu-targets"]


if _cuda_arch_flag:
    _nvcc_flags.append(_cuda_arch_flag)
if pystencils.gpucuda.cudajit.USE_FAST_MATH:
    _nvcc_flags.append('-use_fast_math')


def compile_file(file,
                 use_nvcc=False,
                 nvcc=None,
                 overwrite_destination_file=True,
                 additional_compile_flags=[],
                 openmp=True):
    if 'tensorflow_host_compiler' not in get_compiler_config():
        get_compiler_config()['tensorflow_host_compiler'] = get_compiler_config()['command']
    import tensorflow as tf

    if use_nvcc:
        command_prefix = [nvcc or NVCC_BINARY,
                          '--expt-relaxed-constexpr',
                          '-ccbin',
                          get_compiler_config()['command'],
                          '-Xcompiler',
                          get_compiler_config()['flags'].replace('c++11', 'c++14'),
                          *_nvcc_flags,
                          file,
                          '-x',
                          'cu',
                          '-Xcompiler',
                          _position_independent_flag,
                          _do_not_link_flag,
                          *tf.sysconfig.get_compile_flags(),
                          *_include_flags,
                          *additional_compile_flags,
                          _output_flag]
    else:
        command_prefix = [get_compiler_config()['command'],
                          *(get_compiler_config()['flags']).split(' '),
                          file,
                          _do_not_link_flag,
                          *tf.sysconfig.get_compile_flags(),
                          *_include_flags,
                          *additional_compile_flags,
                          _output_flag]

    # if openmp:
        # command_prefix.insert(2, _output_flag)
    destination_file = f'{file}_{_hash(".".join(command_prefix).encode()).hexdigest()}{_object_file_extension}'

    if not exists(destination_file) or overwrite_destination_file:
        command = command_prefix + [destination_file]
        subprocess.check_call(command, env=_compile_env)

    return destination_file


def compile_sources_and_load(host_sources,
                             cuda_sources=[],
                             additional_compile_flags=[],
                             additional_link_flags=[],
                             compile_only=False,
                             link_cudart=False):

    import tensorflow as tf

    object_files = []
    sources = host_sources + cuda_sources

    def compile(source):
        is_cuda = source in cuda_sources

        if exists(source):
            source_code = read_file(source)
        else:
            source_code = source

        file_extension = '.cu' if is_cuda else '.cpp'
        file_name = join(get_cache_config()['object_cache'],
                         f'{_hash(source_code.encode()).hexdigest()}{file_extension}')
        if not exists(file_name):
            write_file(file_name, source_code)

        object_file = compile_file(file_name,
                                   use_nvcc=is_cuda,
                                   overwrite_destination_file=False,
                                   additional_compile_flags=additional_compile_flags)
        return object_file

    # p_tqdm is just a parallel tqdm
    object_files = p_tqdm.p_umap(compile, sources, desc='Compiling Tensorflow module')

    print('Linking Tensorflow module...')
    module_file = link(object_files,
                       overwrite_destination_file=False,
                       additional_link_flags=additional_link_flags,
                       link_cudart=link_cudart and cuda_sources)
    if not compile_only:
        module = tf.load_op_library(module_file)
        if module:
            print('Loaded Tensorflow module.')
        return module
    else:
        return module_file
