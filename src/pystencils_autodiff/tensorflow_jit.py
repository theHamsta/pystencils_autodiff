# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import subprocess
import sysconfig
from itertools import chain
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
else:
    _do_not_link_flag = "/c"
    _output_flag = '/OUT:'
    _shared_object_flag = '/DLL'
    _include_flags = ['/I' + sysconfig.get_paths()['include'], '/I' + get_pystencils_include_path()]


try:
    import tensorflow as tf

    _tf_compile_flags = tf.sysconfig.get_compile_flags()
    _tf_link_flags = tf.sysconfig.get_link_flags()
except ImportError:
    pass


def link_and_load(object_files, destination_file=None, link_cudart=False, overwrite_destination_file=True):
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
                   *(get_compiler_config()['flags']).split(' '),
                   *object_files,
                   *_tf_link_flags,
                   *_tf_compile_flags,
                   *_include_flags,
                   _shared_object_flag,
                   _output_flag + destination_file]  # /out: for msvc???
        if link_cudart:
            command.append('-lcudart')

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
        return "-arch " + arch
    else:
        return None


_cuda_arch_flag = try_get_cuda_arch_flag()


def compile_file(file, use_nvcc=False, nvcc='nvcc', overwrite_destination_file=True):

    destination_file = file + '.o'
    if use_nvcc:
        command = [nvcc,
                   '--expt-relaxed-constexpr',
                   '-ccbin',
                   get_compiler_config()['command'],
                   *(get_compiler_config()['flags']).split(' '),
                   file,
                   '-x',
                   'cu',
                   '-Xcompiler',
                   '-fPIC',  # TODO: msvc!
                   _do_not_link_flag,
                   *_tf_compile_flags,
                   *_include_flags,
                   _output_flag + destination_file]
        if _cuda_arch_flag:
            command.append(_cuda_arch_flag)
    else:
        command = [get_compiler_config()['command'],
                   *(get_compiler_config()['flags']).split(' '),
                   file,
                   _do_not_link_flag,
                   *_tf_compile_flags,
                   *_include_flags,
                   _output_flag + destination_file]
    if not exists(destination_file) or overwrite_destination_file:
        subprocess.check_call(command)
    return destination_file


def compile_sources_and_load(host_sources, cuda_sources=[]):

    object_files = []

    for source in tqdm(chain(host_sources, cuda_sources), desc='Compiling Tensorflow module...'):
        is_cuda = source in cuda_sources

        if exists(source):
            source_code = read_file(source)
        else:
            source_code = source

        file_extension = '.cu' if is_cuda else '.cpp'
        file_name = join(pystencils.cache.cache_dir, f'{abs(hash(source_code)):x}{file_extension}')
        write_file(file_name, source_code)

        compile_file(file_name, use_nvcc=is_cuda, overwrite_destination_file=False)
        object_files.append(file_name)

    print('Linking Tensorflow module...')
    module = link_and_load(object_files, overwrite_destination_file=False, link_cudart=cuda_sources or False)
    if module:
        print('Loaded Tensorflow module')
    return module
