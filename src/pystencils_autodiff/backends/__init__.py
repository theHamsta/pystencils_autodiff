"""
Backends for operators to support automatic Differentation

Currently, we can use pystencils' JIT compilation to register
a Torch or a Tensorflow operation or we can compile a static
library to be directly loaded into Torch/Tensorflow.
"""

AVAILABLE_BACKENDS = ['tensorflow', 'torch', 'tensorflow_native', 'torch_native']
