from typing import Union

import numpy as np

import pystencils
import pystencils_autodiff
from pystencils_autodiff.lbm.adjoint_boundaryconditions import AdjointBoundaryCondition
from lbmpy.boundaries.boundaryhandling import LatticeBoltzmannBoundaryHandling
from lbmpy.lbstep import LatticeBoltzmannStep
from lbmpy.methods.abstractlbmethod import AbstractLbMethod, LbmCollisionRule
from pystencils import Field
from pystencils.timeloop import TimeLoop


class PdfFieldNotDetectedException(Exception):
    pass


class SimulationResultsTensors:
    def __init__(self, input_pdf_tensor, output_pdf_tensor, output_density_tensor, output_velocity_tensor):
        self.input_pdf_tensor = input_pdf_tensor
        self.output_pdf_tensor = output_pdf_tensor
        self.output_density_tensor = output_density_tensor
        self.output_velocity_tensor = output_velocity_tensor


def _guess_src_dst_field_from_update_rule(update_rule, src_hint, dst_hint):
    # In case of strings, make educated guesses
    if isinstance(src_hint, str):
        src_candidates = [f for f in update_rule.free_fields if f.index_dimensions == 1 and f.index_shape[0] >= 9]
        if len(src_candidates) != 1:
            raise PdfFieldNotDetectedException(
                'Could not guess source PDF field from update rule.' +
                'Please specify the field explicitly in the constructor of AutoDiffLatticeBoltzmannStep!')

    if isinstance(dst_hint, str):
        dst_candidates = [f for f in update_rule.bound_fields if f.index_dimensions == 1 and f.index_shape[0] >= 9]
        if len(dst_candidates) != 1:
            raise PdfFieldNotDetectedException(
                'Could not guess temporary PDF field from update rule.' +
                'Please specify the field explicitly in the constructor of AutoDiffLatticeBoltzmannStep!')

    src_pdf_field = src_hint if isinstance(src_hint, Field) else src_candidates[0]
    dst_pdf_field = dst_hint if isinstance(dst_hint, Field) else dst_candidates[0]

    return src_pdf_field, dst_pdf_field


class AutoDiffLatticeBoltzmannStep(LatticeBoltzmannStep):

    def __init__(self,
                 update_rule: LbmCollisionRule,
                 src_pdf_field: Union[Field, str] = '',
                 tmp_pdf_field: Union[Field, str] = '',
                 time_constant_fields=[],
                 *args,
                 constant_fields=[],
                 optimization={},
                 additional_fields=[],
                 **method_parameters):

        self._adjoint_boundary_conditions = dict()

        src_pdf_field, tmp_pdf_field = _guess_src_dst_field_from_update_rule(update_rule, src_pdf_field, tmp_pdf_field)
        self.pdf_field = src_pdf_field
        self.temporary_field = tmp_pdf_field

        # Field.layout is just spatial
        try:
            real_layout = pystencils.field.get_layout_from_strides(src_pdf_field.strides)
            ndim = len(src_pdf_field.shape)

            if pystencils.field.layout_string_to_tuple('fzyx', dim=ndim) == real_layout:
                layout_string = 'fzyx'
            elif pystencils.field.layout_string_to_tuple('tensorflow', dim=ndim) == real_layout:
                layout_string = 'tensorflow'
            elif pystencils.field.layout_string_to_tuple('zyxf', dim=ndim) == real_layout:
                layout_string = 'zyxf'
            elif pystencils.field.layout_string_to_tuple('c', dim=ndim) == real_layout:
                layout_string = 'c'
            else:
                raise NotImplementedError(
                    "Cannot recognize layout of src_pdf_field")
            optimization.update({'field_layout': layout_string})
        except Exception:
            pass

        optimization.update(
            {'double_precision': src_pdf_field.dtype.numpy_dtype == np.float64})

        super().__init__(*args,
                         update_rule=update_rule,
                         optimization=optimization,
                         pdf_arr_name=src_pdf_field.name,
                         tmp_arr_name=tmp_pdf_field.name,
                         **method_parameters)

        self._target = 'gpu' if self._gpu else 'cpu'
        self._forward_assignments = update_rule
        # TODO: optimization parameters, e.g. target
        self._autodiff = pystencils_autodiff.AutoDiffOp(
            self._forward_assignments,
            "LBM", ghost_layers=self._data_handling.default_ghost_layers,
            target=self._target,
            data_type="double" if src_pdf_field.dtype.numpy_dtype == np.float64 else "float",
            cpu_openmp=True, time_constant_fields=time_constant_fields, constant_fields=constant_fields)

        self._lbmKernels = [
            self._autodiff.get_forward_kernel(is_gpu=self._gpu)]
        self._backwardLbmKernels = [
            self._autodiff.get_backward_kernel(is_gpu=self._gpu)]

        for field in set(additional_fields).union(set(time_constant_fields)):
            if field.name not in self.data_handling.cpu_arrays:
                self._data_handling.add_array(
                    field.name,
                    field.values_per_cell,
                    layout=field.layout,
                    dtype=field.dtype.numpy_dtype,
                    gpu=self._gpu)
            adjoint_field = pystencils_autodiff.AdjointField(field)
            if adjoint_field.name not in self.data_handling.cpu_arrays:
                self._data_handling.add_array(
                    adjoint_field.name,
                    field.values_per_cell,
                    layout=field.layout,
                    dtype=field.dtype.numpy_dtype,
                    gpu=self._gpu)

        for forward, backward in self._autodiff.backward_fields_map.items():
            if backward.name not in self._data_handling.cpu_arrays:
                self._data_handling.add_array_like(
                    backward.name, forward.name, gpu=self._gpu)

        self._backward_boundary_handling = LatticeBoltzmannBoundaryHandling(self.method, self._data_handling,
                                                                            self._backward_tmp_array_name,
                                                                            name="backward_boundary_handling",
                                                                            flag_interface=None,  # TODO
                                                                            target=self._target,
                                                                            openmp=True)

    @property
    def backward_pdf_array_name(self):
        return "diff" + self._tmp_arr_name

    @property
    def _backward_tmp_array_name(self):
        return "diff" + self._pdf_arr_name

    @property
    def forward_assignments(self):
        return self._autodiff.forward_assignments

    @property
    def backward_assignments(self):
        return self._autodiff.backward_assignments

    @property
    def backward_boundary_handling(self):
        return self._backward_boundary_handling

    def set_boundary_including_adjoint(self,
                                       boundary_condition,
                                       slice_obj=None,
                                       mask_callback=None,
                                       mask_array=None,
                                       adjoint_boundary_condition=None):
        self._adjoint_boundary_conditions['domain'] = 'domain'

        if not adjoint_boundary_condition:
            if boundary_condition not in self._adjoint_boundary_conditions:
                self._adjoint_boundary_conditions[boundary_condition] = AdjointBoundaryCondition(
                    boundary_condition,
                    time_constant_fields=self._autodiff.time_constant_fields,
                    constant_fields=self._autodiff.constant_fields)
            adjoint_boundary_condition = self._adjoint_boundary_conditions[boundary_condition]

        self.boundary_handling.set_boundary(
            boundary_condition, slice_obj, mask_callback=mask_callback, mask_array=mask_array)
        self.backward_boundary_handling.set_boundary(
            adjoint_boundary_condition, slice_obj, mask_callback=mask_callback, mask_array=mask_array)
        adjoint_boundary_condition = self._adjoint_boundary_conditions[boundary_condition]

        self.boundary_handling.set_boundary(
            boundary_condition, slice_obj, mask_callback=mask_callback)
        self.backward_boundary_handling.set_boundary(
            adjoint_boundary_condition, slice_obj, mask_callback=mask_callback)

    def create_timestep_op(self, num_time_steps, input_field_to_tensor_dict, backend='tensorflow'):

        def forward_loop(is_cuda=False, **forward_field_names):
            assert is_cuda == (self._target == 'gpu')

            for name in forward_field_names:

                if is_cuda:
                    self.data_handling.gpu_arrays[name] = forward_field_names[name]
                else:
                    self.data_handling.cpu_arrays[name] = pystencils_autodiff._layout_fixer.fix_layout(
                        forward_field_names[name], self.data_handling.fields[name], backend)

            self.run(num_time_steps)
            # pyconrad.imshow(
            #     self.data_handling.cpu_arrays[self.pdf_array_name][..., 0], 'forward')

            if not is_cuda:
                rtn = {f.name: np.ascontiguousarray(
                    self.data_handling.cpu_arrays[f.name]) for f in self._autodiff._forward_output_fields}
                # Due to swapping, valid PDF field will be self._pdf_arr_name instead self._tmp_arr_name
                rtn[self._tmp_arr_name] = np.ascontiguousarray(
                    self.data_handling.cpu_arrays[self._pdf_arr_name])
                return rtn

        def backward_loop(is_cuda=False, **backward_field_names):
            assert is_cuda == (self._target == 'gpu')

            for name in backward_field_names:
                if is_cuda:
                    self.data_handling.gpu_arrays[name] = backward_field_names[name]
                else:
                    try:
                        self.data_handling.cpu_arrays[name] = pystencils_autodiff._layout_fixer.fix_layout(
                            backward_field_names[name], self.data_handling.fields[name], backend)
                    except AttributeError:
                        self.data_handling.cpu_arrays[name] = backward_field_names[name]

            # Set autodiff outputfield to 0
            for f in self._autodiff.backward_output_fields:
                self.data_handling.cpu_arrays[f.name][...] = 0

            self.run_backward(num_time_steps)
            # pyconrad.imshow(
            #     self.data_handling.cpu_arrays[self.backward_pdf_array_name][..., 0], 'backward')

            if not is_cuda:
                rtn = {f.name: np.ascontiguousarray(self.data_handling.cpu_arrays[f.name])
                       for f in self._autodiff._backward_output_fields}
                # Because of swapping the valid pdf field will be self.backward_pdf_array_name
                # Due to swapping, valid PDF field will be self._pdf_arr_name instead self._tmp_arr_name
                rtn[self._backward_tmp_array_name] = np.ascontiguousarray(
                    self.data_handling.cpu_arrays[self.backward_pdf_array_name])
                return rtn

        return self._autodiff.create_tensorflow_op(input_field_to_tensor_dict,
                                                   forward_loop,
                                                   backward_loop,
                                                   backend=backend)

    def create_macroscopic_getter_op(self,
                                     pdf_input_tensor,
                                     force_field_tensor=None,
                                     backend='tensorflow',
                                     **kernel_compilation_kwargs):
        """Creates a Tensorflow Op or a operation for another framework

        Arguments:
            pdf_input_tensor {[type]} -- [description]

        Keyword Arguments:
            force_field_tensor {[type]} -- [description] (default: {None})
            backend {str} -- [description] (default: {'tensorflow'})

        Returns:
            [type] -- [description]
        """

        pdf_field = self._data_handling.fields[self._pdf_arr_name]

        if self.force_data_name:
            assert force_field_tensor is not None, "must set force_field_tensor if using a force model"
            force_field = self._data_handling.fields[self.force_data_name]
        else:
            force_field = None

        getter_eqs, _ = self._create_macroscopic_setter_and_getter_equations()

        op = pystencils_autodiff.AutoDiffOp(
            getter_eqs, self.name + "_SetMacroscopicValues", **kernel_compilation_kwargs)

        return op.create_tensorflow_op({pdf_field: pdf_input_tensor, force_field: force_field_tensor}, backend=backend)

    def create_macroscopic_setter_op(self,
                                     velocity_input_tensor,
                                     density_input_tensor,
                                     force_input_tensor=None,
                                     backend='tensorflow',
                                     **kernel_compilation_kwargs):
        rho_field = self._data_handling.fields[self.density_data_name]
        vel_field = self._data_handling.fields[self.velocity_data_name]

        if self.force_data_name:
            assert force_input_tensor is not None, "must set force_field_tensor if using a force model"
            force_field = self._data_handling.fields[self.force_data_name]
        else:
            force_field = None

        _, setter_eqs = self._create_macroscopic_setter_and_getter_equations()

        op = pystencils_autodiff.AutoDiffOp(
            setter_eqs, self.name + "_SetMacroscopicValues", **kernel_compilation_kwargs)

        rtn_dict = op.create_tensorflow_op(
            {rho_field: density_input_tensor,
             vel_field: velocity_input_tensor,
             force_field: force_input_tensor},
            backend=backend)

        return rtn_dict

    def create_end_to_end_op(self,
                             num_time_steps,
                             velocity_input_tensor,
                             density_input_tensor,
                             additional_fields_to_tensor_map=dict(),
                             force_input_tensor=None,
                             backend='tensorflow',
                             num_times_steps_without_save=0,
                             **kernel_compilation_kwargs,
                             ):
        assert backend == 'tensorflow', 'end-to-end only implemented for tensorflow'
        input_pdf_tensor = self.create_macroscopic_setter_op(
            velocity_input_tensor, density_input_tensor, force_input_tensor, backend, **kernel_compilation_kwargs)

        output_pdf_tensor = input_pdf_tensor
        for _ in range(num_time_steps // (num_times_steps_without_save + 1)):
            additional_fields_to_tensor_map.update({self.pdf_field: output_pdf_tensor})
            output_pdf_tensor = self.create_timestep_op(
                num_times_steps_without_save + 1, additional_fields_to_tensor_map, backend=backend)

        rtn_dict = self.create_macroscopic_getter_op(output_pdf_tensor)
        rtn = SimulationResultsTensors(input_pdf_tensor, output_pdf_tensor,
                                       rtn_dict[self.density_data_name], rtn_dict[self.velocity_data_name])

        return rtn

    def get_backward_time_loop(self):
        self.pre_run()  # make sure GPU arrays are allocated

        fixed_loop = TimeLoop(steps=2)
        fixed_loop.add_pre_run_function(self.pre_run)
        fixed_loop.add_post_run_function(self.post_run)
        fixed_loop.add_single_step_function(self.backward_time_step)

        for t in range(2):
            if len(self._backwardLbmKernels) == 2:  # collide stream
                collide_args = self._data_handling.get_kernel_kwargs(
                    self._backwardLbmKernels[0], **self.kernel_params)
                fixed_loop.add_call(self._backwardLbmKernels[0], collide_args)

                fixed_loop.add_call(self._sync_src if t ==
                                    0 else self._sync_tmp, {})
                self.backward_boundary_handling.add_fixed_steps(
                    fixed_loop, **self.kernel_params)

                stream_args = self._data_handling.get_kernel_kwargs(
                    self._backwardLbmKernels[1], **self.kernel_params)
                fixed_loop.add_call(self._backwardLbmKernels[1], stream_args)
            else:  # stream collide
                stream_collide_args = self._data_handling.get_kernel_kwargs(
                    self._backwardLbmKernels[0], **self.kernel_params)
                fixed_loop.add_call(
                    self._backwardLbmKernels[0], stream_collide_args)
                self.backward_boundary_handling.add_fixed_steps(
                    fixed_loop, **self.kernel_params)
                fixed_loop.add_call(self._sync_src if t ==
                                    0 else self._sync_tmp, {})

            self._data_handling.swap(
                self.backward_pdf_array_name, self._backward_tmp_array_name, self._gpu)
        return fixed_loop

    def backward_time_step(self):
        if len(self._backwardLbmKernels) == 2:  # collide stream
            self._data_handling.run_kernel(
                self._backwardLbmKernels[0], **self.kernel_params)
            self._sync_src()
            self._data_handling.run_kernel(
                self._backwardLbmKernels[1], **self.kernel_params)
            self.backward_boundary_handling(**self.kernel_params)
        else:  # stream collide
            self._data_handling.run_kernel(
                self._backwardLbmKernels[0], **self.kernel_params)
            self.backward_boundary_handling(**self.kernel_params)
            self._sync_src()

        self._data_handling.swap(
            self.backward_pdf_array_name, self._backward_tmp_array_name)

    def run_backward(self, time_steps, show_progressbar=False):
        # TODO: hack
        # Set autodiff outputfield to 0
        for f in set(self._autodiff.backward_output_fields).union(
                pystencils_autodiff.AdjointField(f) for f in self._autodiff._time_constant_fields
        ):
            self.data_handling.cpu_arrays[f.name][...] = 0

        time_loop = self.get_backward_time_loop()
        time_loop.run(time_steps, show_progressbar)

    @property
    def _pre_collision_pdf_symbols(self):
        return self._update_rule.method.pre_collision_pdf_symbols

    @property
    def _post_collision_pdf_symbols(self):
        return self._update_rule.method.post_collision_pdf_symbols

    @property
    def lb_methdod(self) -> AbstractLbMethod:
        return self._update_rule.method
