import sys

import pystencils_autodiff._backport
import pystencils_autodiff.backends  # NOQA
import pystencils_autodiff.tensorflow_jit
from pystencils_autodiff._adjoint_field import AdjointField
from pystencils_autodiff._autodiff import (
    AutoDiffAstPair, AutoDiffBoundaryHandling, AutoDiffOp, DiffModes, create_backward_assignments,
    get_jacobian_of_assignments)
from pystencils_autodiff._field_to_tensors import (  # NOQA
    tf_constant_from_field, tf_placeholder_from_field, tf_scalar_variable_from_field,
    tf_variable_from_field, torch_tensor_from_field)

__all__ = ['backends',
           'AdjointField',
           'get_jacobian_of_assignments',
           'create_backward_assignments',
           'AutoDiffOp',
           'AutoDiffAstPair',
           'tensorflow_jit',
           'tf_constant_from_field', ' tf_placeholder_from_field',
           'tf_scalar_variable_from_field', ' tf_variable_from_field',
           'torch_tensor_from_field',
           'DiffModes',
           'AutoDiffBoundaryHandling']

sys.modules['pystencils.autodiff'] = pystencils_autodiff
sys.modules['pystencils.autodiff.backends'] = pystencils_autodiff.backends
