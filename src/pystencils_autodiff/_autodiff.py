import collections
from enum import Enum
from typing import List

import jinja2
import numpy as np
import sympy as sp

import pystencils as ps
import pystencils.cache
import pystencils_autodiff._layout_fixer
from pystencils.interpolation_astnodes import InterpolatorAccess
from pystencils.math_optimizations import ReplaceOptim, optimize_assignments
from pystencils_autodiff.backends import AVAILABLE_BACKENDS
from pystencils_autodiff.transformations import add_fixed_constant_boundary_handling

REMOVE_CASTS = ReplaceOptim(lambda x: isinstance(x, pystencils.data_types.cast_func), lambda x: x.args[0])
DEFAULT_OP_NAME = "autodiffop"


@pystencils.cache.disk_cache_no_fallback
def _create_backward_assignments_tf_mad(self, diff_fields_prefix):
    """
    Performs the automatic backward differentiation in a more fancy way with write accesses
    like in the forward pass (only flipped).
    It is called "transposed-mode forward-mode algorithmic differentiation" (TF-MAD).

    See this presentation https://autodiff-workshop.github.io/slides/Hueckelheim_nips_autodiff_CNN_PDE.pdf or that
    paper https://www.tandfonline.com/doi/full/10.1080/10556788.2018.1435654?scroll=top&needAccess=true
    for more information
    """

    forward_assignments = self._forward_assignments

    if hasattr(forward_assignments, 'new_without_subexpressions'):
        forward_assignments = forward_assignments.new_without_subexpressions()
    if hasattr(forward_assignments, 'main_assignments'):
        forward_assignments = forward_assignments.main_assignments

    if not hasattr(forward_assignments, 'free_symbols'):
        forward_assignments = ps.AssignmentCollection(
            forward_assignments, [])
    forward_assignments = ps.AssignmentCollection(optimize_assignments(forward_assignments, [
        REMOVE_CASTS
    ]))

    read_field_accesses = sorted([
        a for a in forward_assignments.free_symbols if isinstance(a, ps.Field.Access)]
        + list(forward_assignments.atoms(InterpolatorAccess)), key=str)
    write_field_accesses = sorted([a.lhs for a in forward_assignments], key=lambda x: str(x))
    read_fields = {s.field for s in read_field_accesses}
    write_fields = {s.field for s in write_field_accesses}

    self._forward_read_accesses = read_field_accesses
    self._forward_write_accesses = write_field_accesses
    self._forward_input_fields = sorted(list(read_fields), key=str)
    self._forward_output_fields = sorted(list(write_fields), key=str)

    write_field_accesses = [a.lhs for a in forward_assignments if isinstance(a.lhs, ps.Field.Access)]
    assert write_field_accesses, "No write accesses found"

    # for every field create a corresponding diff field
    diff_read_fields = {f: pystencils_autodiff.AdjointField(f, diff_fields_prefix)
                        for f in read_fields if (f not in self._constant_fields
                                                 and f.name not in self._constant_fields)}
    diff_write_fields = {f: pystencils_autodiff.AdjointField(f, diff_fields_prefix)
                         for f in write_fields}

    assert all(isinstance(w, ps.Field.Access)
               for w in write_field_accesses), \
        "Please check if your assignments are a AssignmentCollection or main_assignments only"

    backward_assignment_dict = collections.OrderedDict()
    # for each output of forward operation
    for _, forward_assignment in enumerate(forward_assignments.main_assignments):
        # we have only one assignment
        diff_write_field = diff_write_fields[forward_assignment.lhs.field]
        diff_write_index = forward_assignment.lhs.index

        # TODO: simplify implementation. use matrix notation like in 'transposed' mode
        for forward_read_field in self._forward_input_fields:
            if forward_read_field in self._constant_fields or forward_read_field.name in self._constant_fields:
                continue
            diff_read_field = diff_read_fields[forward_read_field]

            if diff_read_field.index_dimensions == 0:

                diff_read_field_sum = 0
                for ra in read_field_accesses:
                    if ra.field != forward_read_field:
                        continue  # ignore constant fields in differentiation

                    if isinstance(ra, InterpolatorAccess):
                        out_symbols = sp.symbols(f'o:{len(ra.offsets)}')
                        in_symbols = pystencils.x_vector(len(ra.offsets))
                        maybe_solution = sp.solve(sp.Matrix(ra.offsets) - sp.Matrix(out_symbols), in_symbols)
                        assert maybe_solution, f'Could not solve for {in_symbols} when trying to derive for interpolator'  # noqa
                        inverted_offset = tuple(maybe_solution.values())

                        inverted_offset = [foo.subs({o: s for o, s in zip(out_symbols, in_symbols)})
                                           for foo in inverted_offset]
                        diff_read_field_sum += (sp.diff(forward_assignment.rhs, ra) *
                                                diff_write_field.interpolated_access(inverted_offset))
                    else:
                        # TF-MAD requires flipped stencils
                        inverted_offset = tuple(-v - w for v,
                                                w in zip(ra.offsets, forward_assignment.lhs.offsets))
                        diff_read_field_sum += sp.diff(forward_assignment.rhs, ra) * \
                            diff_write_field[inverted_offset](*diff_write_index)
                if self.time_constant_fields is not None and forward_read_field in self._time_constant_fields:
                    # Accumulate in case of time_constant_fields
                    assignment = ps.Assignment(
                        diff_read_field.center(), diff_read_field.center() + diff_read_field_sum)
                else:
                    # If time dependent, we just need to assign the sum for the current time step
                    assignment = ps.Assignment(
                        diff_read_field.center(), diff_read_field_sum)

                # We can have contributions from multiple forward assignments
                if assignment.lhs in backward_assignment_dict:
                    backward_assignment_dict[assignment.lhs].append(assignment.rhs)
                else:
                    backward_assignment_dict[assignment.lhs] = [assignment.rhs]

            elif diff_read_field.index_dimensions == 1:

                diff_read_field_sum = {}
                for ra in read_field_accesses:
                    if ra.field != forward_read_field:
                        continue  # ignore constant fields in differentiation

                    # TF-MAD requires flipped stencils
                    inverted_offset = tuple(-v - w for v,
                                            w in zip(ra.offsets, write_field_accesses[0].offsets))
                    diff_read_field_sum[ra.index[0]] = diff_read_field_sum.get(
                        ra.index[0], 0) + sp.diff(forward_assignment.rhs, ra) * diff_write_field[inverted_offset]

                for index in diff_read_field_sum.keys():
                    if self.time_constant_fields is not None and forward_read_field in self._time_constant_fields:
                        # Accumulate in case of time_constant_fields
                        assignment = ps.Assignment(
                            diff_read_field.center.at_index(index),
                            diff_read_field.center.at_index(index) + diff_read_field_sum[index])
                    else:
                        # If time dependent, we just need to assign the sum for the current time step
                        assignment = ps.Assignment(
                            diff_read_field.center.at_index(index), diff_read_field_sum[index])

                if assignment.lhs in backward_assignment_dict:
                    backward_assignment_dict[assignment.lhs].append(assignment.rhs)
                else:
                    backward_assignment_dict[assignment.lhs] = [assignment.rhs]
            else:
                raise NotImplementedError()

    backward_assignments = [ps.Assignment(k, sp.Add(*v)) for k, v in backward_assignment_dict.items()]

    try:
        if self._do_common_subexpression_elimination:
            backward_assignments = ps.simp.sympy_cse_on_assignment_list(
                backward_assignments)
    except Exception:
        pass
        # print("Common subexpression elimination failed")
        # print(err)
    main_assignments = [a for a in backward_assignments if isinstance(a.lhs, ps.Field.Access)]
    subexpressions = [a for a in backward_assignments if not isinstance(a.lhs, ps.Field.Access)]
    backward_assignments = ps.AssignmentCollection(main_assignments, subexpressions)

    assert _has_exclusive_writes(backward_assignments), "Backward assignments don't have exclusive writes!"
    self._backward_field_map = {**diff_read_fields, **diff_write_fields}

    return backward_assignments


class AutoDiffBoundaryHandling(str, Enum):
    """
    Strategies for in-kernel boundary handling for AutoDiffOp

    ======= =====================================================
    Variant Effect
    ======= =====================================================
    None    No boundary handling is integrated into the kernels.
            You have to do it yourself to obtain valid gradients.

    ZEROS   Out-of-bound values are assumed to be zero.
            This affects both forward and backward pass.

    VALID   No boundary handling in the forward pass.
            Gradients are calculated correctly in the backward
            pass by assuming zeros when treating out-of-bounds
            regions
    ======= =====================================================
    """
    NONE = None
    ZEROS = 'zeros'
    VALID = 'valid'


class DiffModes(str, Enum):
    """
    Mode of backward differentiation
    (see https://autodiff-workshop.github.io/slides/Hueckelheim_nips_autodiff_CNN_PDE.pdf)
    """
    TRANSPOSED = 'transposed'
    TF_MAD = 'transposed-forward'


class AutoDiffOp:
    """
    Class to generate code for forward and backward kernels for automatic
    differentiation operators.
    """

    _REPR_TEMPLATE = jinja2.Template(
        """Forward:
    {{ forward_assignments | indent(4) }}
Backward:
    {{ backward_assignments | indent(4) }}
""")

    def __init__(self,
                 forward_assignments: List[ps.Assignment],
                 op_name: str = DEFAULT_OP_NAME,
                 boundary_handling: AutoDiffBoundaryHandling = None,
                 time_constant_fields: List[ps.Field] = None,
                 constant_fields: List[ps.Field] = [],
                 diff_fields_prefix='diff',  # TODO: remove!
                 do_common_subexpression_elimination=True,
                 diff_mode=DiffModes.TF_MAD,
                 backward_assignments=None,
                 **kwargs):
        diff_mode = DiffModes(diff_mode)
        assert diff_mode in DiffModes, f"Please select a valid differentiation mode: {DiffModes.__members__.values()}"
        self._additional_symbols = []

        if 'target' in kwargs:
            assert kwargs['target'].lower() in [
                'cpu', 'gpu'], "AutoDiffOp always supports both cpu and gpu"
            del kwargs['target']

        main_assignments = [a for a in forward_assignments if isinstance(a.lhs, ps.Field.Access)]
        subexpressions = [a for a in forward_assignments if not isinstance(a.lhs, ps.Field.Access)]
        forward_assignments = ps.AssignmentCollection(main_assignments, subexpressions)

        if boundary_handling == AutoDiffBoundaryHandling.VALID:
            raise NotImplementedError('there seems to be still a bug with valid. -> Use "zeros"')

        self._forward_assignments = forward_assignments
        self._backward_assignments = None
        self._constant_fields = constant_fields
        self._constant_fields += ['indexVector']
        self._time_constant_fields = time_constant_fields
        self._kwargs = kwargs
        self.op_name = op_name
        self._forward_ast_cpu = None
        self._backward_ast_cpu = None
        self._forward_ast_gpu = None
        self._backward_ast_gpu = None
        self._forward_kernel_cpu = None
        self._backward_kernel_cpu = None
        self._forward_kernel_gpu = None
        self._backward_kernel_gpu = None
        self._do_common_subexpression_elimination = do_common_subexpression_elimination
        self._boundary_handling = boundary_handling

        if backward_assignments:
            self._forward_assignments = forward_assignments
            self._forward_read_accesses = None
            self._forward_write_accesses = None
            self._forward_input_fields = list(sorted(forward_assignments.free_fields, key=lambda x: str(x)))
            self._forward_output_fields = list(sorted(forward_assignments.bound_fields, key=lambda x: str(x)))
            self._backward_assignments = backward_assignments
            self._backward_input_fields = list(sorted(backward_assignments.free_fields, key=lambda x: str(x)))
            self._backward_output_fields = list(sorted(backward_assignments.bound_fields, key=lambda x: str(x)))
        else:
            # if no_caching:
            if diff_mode == 'transposed':
                self._create_backward_assignments(diff_fields_prefix)
            elif diff_mode == 'transposed-forward':
                self._backward_assignments = None
                self._backward_field_map = None
                backward_assignments = _create_backward_assignments_tf_mad(self, diff_fields_prefix)
                self._backward_assignments = backward_assignments

                self._forward_assignments = forward_assignments
                self._forward_read_accesses = None
                self._forward_write_accesses = None
                self._forward_input_fields = list(sorted(forward_assignments.free_fields, key=lambda x: str(x)))
                self._forward_output_fields = list(sorted(forward_assignments.bound_fields, key=lambda x: str(x)))
                self._backward_assignments = backward_assignments
                self._backward_field_map = None
                self._backward_input_fields = list(sorted(backward_assignments.free_fields, key=lambda x: str(x)))
                self._backward_output_fields = list(sorted(backward_assignments.bound_fields, key=lambda x: str(x)))
            else:
                raise NotImplementedError()
            # else:
            #   # self.backward_assignments = create_backward_assignments(forward_assignments,
            #                                                           # diff_fields_prefix,
            #                                                           # time_constant_fields,
            #                                                           # constant_fields,
            #                                                           # diff_mode=diff_mode,
                # do_common_subexpression_elimination=do_common_subexpression_elimination)

    def __hash__(self):
        return hash((str(self.forward_assignments), str(self.backward_assignments), str(self.constant_fields)))

    def __repr__(self):
        return self._REPR_TEMPLATE.render(forward_assignments=str(self.forward_assignments),
                                          backward_assignments=str(self.backward_assignments))

    def __str__(self):
        return self.__repr__()

    def __setstate__(self, state):
        forward_assignments = state['forward_assignments']
        backward_assignments = state['backward_assignments']
        self._forward_assignments = forward_assignments
        self._forward_read_accesses = None
        self._forward_write_accesses = None
        self._forward_input_fields = list(forward_assignments.free_fields)
        self._forward_output_fields = list(forward_assignments.bound_fields)
        self._backward_assignments = backward_assignments
        self._backward_input_fields = list(backward_assignments.free_fields)
        self._backward_output_fields = list(backward_assignments.bound_fields)
        self._forward_ast_cpu = None
        self._backward_ast_cpu = None
        self._forward_ast_gpu = None
        self._backward_ast_gpu = None
        self._forward_kernel_cpu = None
        self._backward_kernel_cpu = None
        self._forward_kernel_gpu = None
        self._backward_kernel_gpu = None
        self._constant_fields = None
        self._time_constant_fields = None
        self._kwargs = state['kwargs']
        self.op_name = ''
        self._forward_ast_cpu = None
        self._backward_ast_cpu = None
        self._forward_ast_gpu = None
        self._backward_ast_gpu = None
        self._forward_kernel_cpu = None
        self._backward_kernel_cpu = None
        self._forward_kernel_gpu = None
        self._backward_kernel_gpu = None
        self._do_common_subexpression_elimination = None
        self._boundary_handling = 'ZEROS'

    def __getstate__(self):
        return {'forward_assignments': self.forward_assignments,
                'backward_assignments': self.backward_assignments,
                'kwargs': self._kwargs}

    def _create_backward_assignments(self, diff_fields_prefix):
        """
        Performs automatic differentiation in the traditional adjoint/tangent way.
        Forward writes become backward reads and vice-versa. This can lead to problems when
        parallel reads lead to parallel writes, and therefore to race conditions.
        Therefore, theres is also _create_backward_assignments_tf_mad that
        can circumvent that problem in the case of stencils that have only one output
        """

        forward_assignments = self._forward_assignments

        if hasattr(forward_assignments, 'new_without_subexpressions'):
            forward_assignments = forward_assignments.new_without_subexpressions()
        if hasattr(forward_assignments, 'main_assignments'):
            forward_assignments = forward_assignments.main_assignments

        if not hasattr(forward_assignments, 'free_symbols'):
            forward_assignments = ps.AssignmentCollection(
                forward_assignments, [])

        read_field_accesses = [
            a for a in forward_assignments.free_symbols if isinstance(a, ps.Field.Access)]
        write_field_accesses = [a.lhs for a in forward_assignments]

        assert all(isinstance(w, ps.Field.Access) for w in write_field_accesses), \
            "Please assure that you only assign to fields in your main_assignments!"

        read_fields = {s.field for s in read_field_accesses}
        write_fields = {s.field for s in write_field_accesses}

        # for every field create a corresponding diff field
        diff_read_fields = {f: pystencils_autodiff.AdjointField(f, diff_fields_prefix)
                            for f in read_fields}
        diff_write_fields = {f: pystencils_autodiff.AdjointField(f, diff_fields_prefix)
                             for f in write_fields}

        # Translate field accesses from standard to diff fields
        diff_read_field_accesses = [diff_read_fields[fa.field][fa.offsets](*fa.index)
                                    for fa in read_field_accesses]
        diff_write_field_accesses = [diff_write_fields[fa.field][fa.offsets](*fa.index)
                                     for fa in write_field_accesses]

        backward_assignments = []
        for lhs, read_access in zip(diff_read_field_accesses, read_field_accesses):
            # don't differentiate for constant fields
            if read_access.field in self._constant_fields or read_access.field.name in self._constant_fields:
                continue

            rhs = sp.Matrix(sp.Matrix([e.rhs for e in forward_assignments])).diff(
                read_access).transpose() * sp.Matrix(diff_write_field_accesses)
            assert rhs.shape == (1, 1)
            rhs = rhs[0, 0]

            # if field is constant over we time we can accumulate in assignment
            if self.time_constant_fields is not None and read_access.field in self._time_constant_fields:
                backward_assignments.append(ps.Assignment(lhs, lhs + rhs))
            else:
                backward_assignments.append(ps.Assignment(lhs, rhs))

        try:
            if self._do_common_subexpression_elimination:
                backward_assignments = ps.simp.sympy_cse_on_assignment_list(
                    backward_assignments)
        except Exception:
            pass

            # print("Common subexpression elimination failed")
            # print(err)
        main_assignments = [a for a in backward_assignments if isinstance(a.lhs, ps.Field.Access)]
        subexpressions = [a for a in backward_assignments if not isinstance(a.lhs, ps.Field.Access)]
        backward_assignments = ps.AssignmentCollection(main_assignments, subexpressions)
        assert _has_exclusive_writes(backward_assignments), "Backward assignments don't have exclusive writes." + \
            " You should consider using 'transposed-forward' mode for resolving those conflicts"

        self._forward_read_accesses = read_field_accesses
        self._forward_write_accesses = write_field_accesses
        self._forward_input_fields = list(read_fields)
        self._forward_output_fields = list(write_fields)
        self._backward_assignments = backward_assignments
        self._backward_field_map = {**diff_read_fields, **diff_write_fields}
        self._backward_input_fields = [
            self._backward_field_map[f] for f in self._forward_output_fields]
        self._backward_output_fields = [
            self._backward_field_map[f] for f in self._forward_input_fields]

    @property
    def forward_assignments(self):
        return self._forward_assignments

    @property
    def backward_assignments(self):
        return self._backward_assignments

    def get_forward_kernel(self, is_gpu):
        if is_gpu:
            return self.forward_kernel_gpu
        else:
            return self.forward_kernel_cpu

    def get_backward_kernel(self, is_gpu):
        if is_gpu:
            return self.backward_kernel_gpu
        else:
            return self.backward_kernel_cpu

    def jacobian(self):
        """Calculates the Jacobian of the forward_assignments with respect to forward read accesses"""
        return get_jacobian_of_assignments(self._forward_assignments, self._forward_read_accesses)

    @property
    def forward_write_accesses(self):
        return self._forward_write_accesses

    @property
    def forward_read_accesses(self):
        return self._forward_read_accesses

    @property
    def backward_write_accesses(self):
        return [a.rhs for a in self.backward_assignments if isinstance(a, ps.Field.Access)]

    @property
    def backward_read_accesses(self):
        return [a for a in self.backward_assignments.free_symbols if isinstance(a, ps.Field.Access)]

    @property
    def forward_ast_cpu(self):
        if not self._forward_ast_cpu:
            if self._boundary_handling == AutoDiffBoundaryHandling.ZEROS:
                self._forward_assignments = add_fixed_constant_boundary_handling(self._forward_assignments)
                ghost_layers = 0
            else:
                ghost_layers = None
            self._forward_ast_cpu = ps.create_kernel(self._forward_assignments,
                                                     ghost_layers=ghost_layers,
                                                     **self._kwargs)

        self._forward_ast_cpu.function_name = self.op_name + '_forward_cpu'
        return self._forward_ast_cpu

    @property
    def forward_ast_gpu(self):
        if not self._forward_ast_gpu:
            if self._boundary_handling == AutoDiffBoundaryHandling.ZEROS:
                self._forward_assignments = add_fixed_constant_boundary_handling(self._forward_assignments)
                ghost_layers = 0
            else:
                ghost_layers = None
            self._forward_ast_gpu = ps.create_kernel(self._forward_assignments,
                                                     ghost_layers=ghost_layers,
                                                     target='gpu',
                                                     **self._kwargs)

        self._forward_ast_gpu.function_name = self.op_name + '_forward_gpu'
        return self._forward_ast_gpu

    @property
    def backward_ast_cpu(self):
        assert self._backward_assignments, 'No backward assignments!'
        if not self._backward_ast_cpu:
            if (self._boundary_handling == AutoDiffBoundaryHandling.ZEROS or
                    self._boundary_handling == AutoDiffBoundaryHandling.VALID):
                self._backward_assignments = add_fixed_constant_boundary_handling(self._backward_assignments)
                ghost_layers = 0
            else:
                ghost_layers = None
            self._backward_ast_cpu = ps.create_kernel(self._backward_assignments,
                                                      ghost_layers=ghost_layers,
                                                      target='cpu',
                                                      **self._kwargs)
        self._backward_ast_cpu.function_name = self.op_name + '_backward_cpu'
        return self._backward_ast_cpu

    @property
    def backward_ast_gpu(self):
        assert self._backward_assignments, 'No backward assignments!'
        if not self._backward_ast_gpu:
            if (self._boundary_handling == AutoDiffBoundaryHandling.ZEROS or
                    self._boundary_handling == AutoDiffBoundaryHandling.VALID):
                self._backward_assignments = add_fixed_constant_boundary_handling(self._backward_assignments)
                ghost_layers = 0
            else:
                ghost_layers = None
            self._backward_ast_gpu = ps.create_kernel(self._backward_assignments,
                                                      ghost_layers=ghost_layers,
                                                      target='gpu',
                                                      **self._kwargs)
        self._backward_ast_gpu.function_name = self.op_name + '_backward_gpu'
        return self._backward_ast_gpu

    @property
    def forward_kernel_cpu(self):
        if not self._forward_kernel_cpu:
            self._forward_kernel_cpu = self.forward_ast_cpu.compile()
        return self._forward_kernel_cpu

    @property
    def forward_kernel_gpu(self):
        if not self._forward_kernel_gpu:
            self._forward_kernel_gpu = self.forward_ast_gpu.compile()
        return self._forward_kernel_gpu

    @property
    def backward_kernel_cpu(self):
        if not self._backward_kernel_cpu:
            self._backward_kernel_cpu = self.backward_ast_cpu.compile()
        return self._backward_kernel_cpu

    @property
    def backward_kernel_gpu(self):
        if not self._backward_kernel_gpu:
            self._backward_kernel_gpu = self.backward_ast_gpu.compile()
        return self._backward_kernel_gpu

    @property
    def backward_input_fields(self):
        return self._backward_input_fields

    @property
    def backward_output_fields(self):
        return self._backward_output_fields

    @property
    def backward_fields(self):
        return self._backward_output_fields + self._backward_input_fields

    @property
    def forward_fields(self):
        return self._forward_output_fields + self._forward_input_fields

    @property
    def forward_input_fields(self):
        return self._forward_input_fields

    @property
    def forward_output_fields(self):
        return self._forward_output_fields

    def create_forward_kernel(self, *args, **kwargs):
        return ps.create_kernel(
            self._forward_assignments, *args, **kwargs).compile()

    def create_backward_kernel(self, *args, **kwargs):
        return ps.create_kernel(
            self._backward_assignments, *args, **kwargs).compile()

    @property
    def constant_fields(self):
        return self._constant_fields

    @property
    def time_constant_fields(self):
        return self._time_constant_fields

    def create_torch_op(self, *args, **kwargs):
        return self.create_tensorflow_op(*args, backend='torch_native', **kwargs)

    def create_tensorflow_op(self,
                             inputfield_tensor_dict={},
                             forward_loop=None,
                             backward_loop=None,
                             use_cuda=True,
                             backend='tensorflow'):
        """
        Creates custom differentiable Tensorflow Op from assignments.
        Will return either a single output tensor or a OrderedDict[field_name -> tf.Tensor] in case of multiple outputs
        """
        backend = backend.lower()
        assert backend in AVAILABLE_BACKENDS, "\"{}\" is not a valid backend. Available backends: {}".format(
            backend, AVAILABLE_BACKENDS)

        additional_fields = [f for f in inputfield_tensor_dict.keys(
        ) if f not in self._forward_input_fields]

        for f in additional_fields:
            if f and isinstance(f, ps.Field):
                f_adjoint = ps.autodiff.AdjointField(f)
                self._forward_input_fields.append(f)
                self._backward_output_fields.append(f_adjoint)
                self._backward_field_map[f] = f_adjoint

        if not forward_loop:
            # raise ValueError('forward_loop == None is not (yet) implemented')

            def forward_function(**kwargs):
                for field in self.forward_input_fields:
                    kwargs[field.name] = pystencils_autodiff._layout_fixer.fix_layout(
                        kwargs[field.name], field, backend)
                # TODO: check dangerous function `as_strided`
                for s in self._additional_symbols:
                    kwargs[s.name] = getattr(self, s.name)
                rtn = {f.name: np.lib.stride_tricks.as_strided(np.zeros(f.shape,
                                                                        dtype=f.dtype.numpy_dtype),
                                                               f.shape,
                                                               [f.dtype.numpy_dtype.itemsize * a for a in f.strides])
                       for f in self.forward_output_fields}

                kwargs.update(rtn)
                self.forward_kernel_cpu(**kwargs)
                return rtn

            forward_loop = forward_function

        if not backward_loop:
            # raise ValueError('backward_loop == None is not (yet) implemented')

            def backward_function(**kwargs):
                for field in self.backward_input_fields + self.forward_input_fields:
                    kwargs[field.name] = pystencils_autodiff._layout_fixer.fix_layout(
                        kwargs[field.name], field, backend)
                for s in self._additional_symbols:
                    kwargs[s.name] = getattr(self, s.name)

                rtn = {f.name: np.lib.stride_tricks.as_strided(np.zeros(f.shape,
                                                                        dtype=f.dtype.numpy_dtype),
                                                               f.shape,
                                                               [f.dtype.numpy_dtype.itemsize * a for a in f.strides])
                       for f in self.backward_output_fields}

                kwargs.update(rtn)

                self.backward_kernel_cpu(**kwargs)
                return rtn

            backward_loop = backward_function

        if backend == 'tensorflow_native':
            import pystencils_autodiff.backends._tensorflow
            op = pystencils_autodiff.backends._tensorflow.native_tensorflowop_from_autodiffop(self, use_cuda)
        elif backend == 'torch':
            import pystencils_autodiff.backends._pytorch
            op = pystencils_autodiff.backends._pytorch.create_autograd_function(
                self, inputfield_tensor_dict, forward_loop, backward_loop)
        elif backend == 'torch_native':
            import pystencils_autodiff.backends._torch_native
            op = pystencils_autodiff.backends._torch_native.create_autograd_function(
                self, use_cuda, op_name=self.op_name if self.op_name != DEFAULT_OP_NAME else None)
        elif backend == 'tensorflow':
            import pystencils_autodiff.backends._tensorflow
            op = pystencils_autodiff.backends._tensorflow.tensorflowop_from_autodiffop(
                self, inputfield_tensor_dict, forward_loop, backward_loop)
        else:
            raise NotImplementedError()

        if backend == 'tensorflow':
            if len(op) == 1:
                return op[0]
            else:
                rtn = collections.OrderedDict()
                for field, tensor in zip(self.forward_output_fields, op):
                    if backend == 'tensorflow' and field.has_fixed_shape:
                        tensor.set_shape(field.shape)
                    rtn[field.name] = tensor
                return rtn
        else:
            return op


@pystencils.cache.disk_cache_no_fallback
def create_backward_assignments(forward_assignments,
                                diff_fields_prefix="diff",
                                time_constant_fields=[],
                                constant_fields=[],
                                diff_mode=DiffModes.TF_MAD,
                                do_common_sub_expression_elimination=True):
    auto_diff = AutoDiffOp(forward_assignments,
                           diff_fields_prefix=diff_fields_prefix,
                           time_constant_fields=time_constant_fields,
                           constant_fields=constant_fields,
                           diff_mode=diff_mode,
                           do_common_subexpression_elimination=do_common_sub_expression_elimination,
                           no_chaching=True
                           )
    backward_assignments = auto_diff.backward_assignments

    return backward_assignments


class AutoDiffAstPair:
    """
    A pair of ASTs of forward and backward kernel.

    Just needed, if compilation from AssignmentCollection is not sufficient and you want to manipulate the ASTs
    """

    def __init__(self, forward_ast, backward_ast, compilation_target='cpu'):
        self.forward_ast = forward_ast
        self.backward_ast = backward_ast
        self._target = compilation_target
        self._forward_kernel = self.forward_ast.compile(target=self._target)
        self._backward_kernel = None

    def backward(self, *args, **kwargs):
        if not self._backward_kernel:
            self._backward_kernel = self.backward_ast.compile(target=self._target)

        return self._backward_kernel(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self._forward_kernel(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _has_exclusive_writes(assignment_collection):
    """
    Simple check for exclusive (non-overlapping) writes.
    I.e. AssignmentCollection can be executed safely in parallel without caring about race conditions.
    No writes on same spatial location (considering all possible shifts).

    The checked condition might be violated if using DiffModes.TRANSPOSED
    """

    assignments = assignment_collection.main_assignments
    write_field_accesses = [a.lhs for a in assignments if isinstance(a.lhs, ps.Field.Access)]

    exclusive_writes = set()
    for a in write_field_accesses:

        if (a.field, a.index) in exclusive_writes:
            return False
        else:
            exclusive_writes.add((a.field, a.index))

    return True


def get_jacobian_of_assignments(assignments, diff_variables):
    """
    Calculates the Jacobian of iterable of assignments wrt. diff_variables

    Arguments:
        assignments (List[pystencils.Assignment]): A collection of assignments or a AssignmentCollection
        diff_variables (List[sympy.Symbol]): List of variables used to differentiate

    Returns:
        sp.Matrix -- Jacobian of statements
    """

    if hasattr(assignments, 'main_assignments'):
        assignments = assignments.main_assignments

    rhs = sp.Matrix([e.rhs for e in assignments])
    return rhs.jacobian(diff_variables)

    rhs = sp.Matrix([e.rhs for e in assignments])
    return rhs.jacobian(diff_variables)
