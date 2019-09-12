# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import json
import subprocess
import sysconfig
from os.path import exists, join

from tqdm import tqdm

import pystencils
from pystencils.cpu.cpujit import get_cache_config, get_compiler_config, get_pystencils_include_path
from pystencils_autodiff._file_io import read_file, write_file

# TODO: msvc
if get_compiler_config()['os'] != 'windows':
    _shared_object_flag = '-shared'
    _output_flag = '-o'
    _include_flags = ['-I' + sysconfig.get_paths()['include'], '-I' + get_pystencils_include_path()]
    _do_not_link_flag = "-c"
    _position_independent_flag = "-fPIC"
else:
    _do_not_link_flag = "/c"
    _output_flag = '/OUT:'
    _shared_object_flag = '/DLL'
    _include_flags = ['/I' + sysconfig.get_paths()['include'], '/I' + get_pystencils_include_path()]
    _position_independent_flag = "/DTHIS_FLAG_DOES_NOTHING"


try:
    import tensorflow as tf

    _tf_compile_flags = tf.sysconfig.get_compile_flags()
    _tf_link_flags = tf.sysconfig.get_link_flags()
except ImportError:
    pass


def link_and_load(object_files, destination_file=None, overwrite_destination_file=True):
    """Compiles given :param:`source_file` to a Tensorflow shared Library.

    .. warning::

        This functions performs no caching. Only call when your source_files changed!

    :sources_file: Files to compile
    :destination_file: Optional destination path and filename for shared object.
    :returns: Object containing all Tensorflow Ops in that shared library.

    """

    if not destination_file:
        destination_file = join(get_cache_config()['object_cache'], f"{abs(hash(tuple(object_files))):x}.so")

    if not exists(destination_file) or overwrite_destination_file:
        command = [get_compiler_config()['command'],
                   _position_independent_flag,
                   *object_files,
                   *_tf_link_flags,
                   *_include_flags,
                   _shared_object_flag,
                   _output_flag,
                   destination_file]  # /out: for msvc???

        subprocess.check_call(command)

    lib = tf.load_op_library(destination_file)
    return lib


def try_get_cuda_arch_flag():
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


def compile_file(file, use_nvcc=False, nvcc='nvcc', overwrite_destination_file=True):
    if 'tensorflow_host_compiler' not in get_compiler_config():
        get_compiler_config()['tensorflow_host_compiler'] = get_compiler_config()['command']
        write_file(pystencils.cpu.cpujit.get_configuration_file_path(), json.dumps(pystencils.cpu.cpujit._config))

    destination_file = file + '.o'
    if use_nvcc:
        command = [nvcc,
                   '--expt-relaxed-constexpr',
                   '-ccbin',
                   get_compiler_config()['tensorflow_host_compiler'],
                   '-Xcompiler',
                   get_compiler_config()['flags'].replace('c++11', 'c++14'),
                   *_nvcc_flags,
                   file,
                   '-x',
                   'cu',
                   '-Xcompiler',
                   _position_independent_flag,
                   _do_not_link_flag,
                   *_tf_compile_flags,
                   *_include_flags,
                   _output_flag,
                   destination_file]
    else:
        command = [get_compiler_config()['command'],
                   *(get_compiler_config()['flags']).split(' '),
                   file,
                   _do_not_link_flag,
                   *_tf_compile_flags,
                   *_include_flags,
                   _output_flag,
                   destination_file]

    if not exists(destination_file) or overwrite_destination_file:
        subprocess.check_call(command)
    return destination_file


def compile_sources_and_load(host_sources, cuda_sources=[]):

    object_files = []

    for source in tqdm(host_sources + cuda_sources, desc='Compiling Tensorflow module'):
        is_cuda = source in cuda_sources

        if exists(source):
            source_code = read_file(source)
        else:
            source_code = source

        file_extension = '.cu' if is_cuda else '.cpp'
        file_name = join(pystencils.cache.cache_dir, f'{abs(hash(source_code)):x}{file_extension}')
        write_file(file_name, source_code)

        compile_file(file_name, use_nvcc=is_cuda, overwrite_destination_file=False)
        object_files.append(file_name + '.o')

    print('Linking Tensorflow module...')
    module = link_and_load(object_files, overwrite_destination_file=False)
    if module:
        print('Loaded Tensorflow module.')
    return module
