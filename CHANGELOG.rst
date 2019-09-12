=========
Changelog
=========

0.2.2
-----
 * Add possibility to overwrite nvcc arch for Tensorflow compilation: PYSTENCILS_TENSORFLOW_NVCC_ARCH
 * Add possibility to compile Tensorflow module without loading

0.2.1
-----
 * Bugfix: tensorflow_jit had erroneous code for writing to pystencils' config file

0.2.0
-----
 * Compilation of Torch/Tensorflow/pybind11 modules with an AST (instead of pure jinja as before)

