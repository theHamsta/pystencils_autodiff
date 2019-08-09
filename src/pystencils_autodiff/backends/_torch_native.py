import os
import uuid
from itertools import chain
from os.path import dirname, isdir, isfile, join

import jinja2
import torch
from appdirs import user_cache_dir

import pystencils
import pystencils_autodiff
import pystencils_autodiff.backends._pytorch
from pystencils.astnodes import FieldShapeSymbol
from pystencils.backends.cbackend import generate_c
from pystencils.backends.cuda_backend import CudaSympyPrinter, generate_cuda
from pystencils.cpu.kernelcreation import create_kernel
from pystencils.gpucuda.kernelcreation import create_cuda_kernel
from pystencils_autodiff.backends._pytorch import numpy_dtype_to_torch


def _read_file(file):
    with open(file, 'r') as f:
        return f.read()


def _write_file(filename, content):
    with open(filename, 'w') as f:
        return f.write(content)


def generate_torch(destination_folder,
                   autodiff: pystencils_autodiff.AutoDiffOp,
                   is_cuda,
                   dtype,
                   forward_ast=None,
                   backward_ast=None):
    shape = autodiff.forward_output_fields[0].spatial_shape
    operation_hash = abs(hash(autodiff) + hash(shape) + hash(str(dtype)))
    operation_string = "{}_native_{}_{}_{:x}".format(
        autodiff.op_name, 'cuda' if is_cuda else 'cpu', 'x'.join(str(s) for s in shape), operation_hash)

    cpp_file = join(destination_folder, operation_string + '.cpp')
    cuda_kernel_file = join(destination_folder, operation_string + '.cu')

    required_files = [cpp_file, cuda_kernel_file] if is_cuda else [cpp_file]

    if not all(isfile(x) for x in required_files):
        generate_ast = create_cuda_kernel if is_cuda else create_kernel
        generate_code = generate_cuda if is_cuda else generate_c

        if not forward_ast:
            forward_ast = generate_ast(autodiff.forward_assignments.all_assignments)
        if not backward_ast:
            backward_ast = generate_ast(autodiff.backward_assignments.all_assignments)

        forward_ast.subs({s: FieldShapeSymbol(
            [autodiff.forward_output_fields[0].name], s.coordinate) for s in forward_ast.atoms(FieldShapeSymbol)})
        backward_ast.subs({s: FieldShapeSymbol(
            [autodiff.backward_output_fields[0].name], s.coordinate) for s in backward_ast.atoms(FieldShapeSymbol)})
        # backward_ast.subs({s: FieldStrideSymbol(
        # autodiff.forward_input_fields[0].name, s.coordinate) for s in forward_ast.atoms(FieldStrideSymbol)})

        forward_code = generate_code(forward_ast.body).replace(
            'float *', 'scalar_t *').replace('double *', 'scalar_t *')
        backward_code = generate_code(backward_ast.body).replace(
            'float *', 'scalar_t *').replace('double *', 'scalar_t *')

        if is_cuda:
            printer = CudaSympyPrinter()
            block_and_thread_numbers = forward_ast.indexing.call_parameters(shape)
            forward_block = ', '.join(printer.doprint(i) for i in block_and_thread_numbers['block'])
            forward_grid = ', '.join(printer.doprint(i) for i in block_and_thread_numbers['grid'])
            backward_shape = autodiff.backward_output_fields[0].spatial_shape
            block_and_thread_numbers = backward_ast.indexing.call_parameters(backward_shape)
            backward_block = ', '.join(printer.doprint(i) for i in block_and_thread_numbers['block'])
            backward_grid = ', '.join(printer.doprint(i) for i in block_and_thread_numbers['grid'])
            cuda_globals = pystencils.backends.cbackend.get_global_declarations(forward_ast) | \
                pystencils.backends.cbackend.get_global_declarations(backward_ast)
            cuda_globals = [generate_cuda(g) for g in cuda_globals]
        else:
            backward_block = forward_block = "INVALID"
            backward_grid = forward_grid = "INVALID"
            cuda_globals = ""

        render_dict = {
            "forward_tensors": [f for f in autodiff.forward_fields],
            "forward_input_tensors": [f for f in autodiff.forward_input_fields],
            "forward_output_tensors": [f for f in autodiff.forward_output_fields],
            "backward_tensors": [f for f in autodiff.backward_fields + autodiff.forward_input_fields],
            "backward_input_tensors": [f for f in autodiff.backward_input_fields],
            "backward_output_tensors": [f for f in autodiff.backward_output_fields],
            "forward_kernel": forward_code,
            "backward_kernel": backward_code,
            "dimensions": range(autodiff.forward_fields[0].spatial_dimensions),
            "kernel_name": operation_string,
            "forward_threads": "{" + forward_block + "}",
            "forward_blocks": "{" + forward_grid + "}",
            "backward_threads": "{" + backward_block + "}",
            "backward_blocks": "{" + backward_grid + "}",
            "cuda_globals": cuda_globals,
            "dtype": pystencils.data_types.BasicType(dtype)
        }

        if is_cuda:
            template_string_cpp = _read_file(join(dirname(__file__),
                                                  'torch_native_cuda.tmpl.cpp'))
            template = jinja2.Template(template_string_cpp)
            output = template.render(render_dict)
            _write_file(join(destination_folder, operation_string + '.cpp'), output)

            template_string = _read_file(join(dirname(__file__), 'torch_native_cuda.tmpl.cu'))
            template = jinja2.Template(template_string)
            output = template.render(render_dict)
            _write_file(join(destination_folder, operation_string + '.cu'), output)
        else:
            template_string_cpp = _read_file(join(dirname(__file__),
                                                  'torch_native_cpu.tmpl.cpp'))
            template = jinja2.Template(template_string_cpp)
            output = template.render(render_dict)
            _write_file(join(destination_folder, operation_string + '.cpp'), output)

    from torch.utils.cpp_extension import load
    compiled_operation = load(operation_string, required_files, verbose=True,
                              extra_cuda_cflags=[] if is_cuda else [])
    compiled_operation.code = output
    return compiled_operation


def create_autograd_function(autodiff_obj, inputfield_to_tensor_dict, forward_loop=None, backward_loop=None):
    if forward_loop is None:
        assert backward_loop is None
        is_cuda = all(t.is_cuda for t in inputfield_to_tensor_dict.values())
        assert all(t.is_cuda for t in inputfield_to_tensor_dict.values()) or \
            all(not t.is_cuda for t in inputfield_to_tensor_dict.values()), "All tensor should be on GPU or all on CPU"
        dtype = pystencils_autodiff.backends._pytorch.torch_dtype_to_numpy(
            list(inputfield_to_tensor_dict.values())[0].dtype)

        cache_dir = user_cache_dir('pystencils')
        if not isdir(cache_dir):
            os.mkdir(cache_dir)
        # TODO: create function and stuff

        compiled_operation = generate_torch(cache_dir, autodiff_obj, is_cuda, dtype)
        field_to_tensor_dict = inputfield_to_tensor_dict
        # Allocate output tensor for forward and backward pass
        for field in chain(autodiff_obj.forward_output_fields, autodiff_obj.backward_output_fields):
            field_to_tensor_dict[field] = torch.zeros(
                *field.shape,
                dtype=numpy_dtype_to_torch(field.dtype.numpy_dtype),
                device=list(inputfield_to_tensor_dict.values())[0].device)

        def forward(self):
            self.saved = {f: field_to_tensor_dict[f] for f in chain(
                autodiff_obj.forward_input_fields, autodiff_obj.backward_output_fields)}
            compiled_operation.forward(**{f.name: field_to_tensor_dict[f] for f in autodiff_obj.forward_fields})
            return tuple(field_to_tensor_dict[f] for f in autodiff_obj.forward_output_fields)

        def backward(self, *grad_outputs):
            self.saved.update({f.name: grad_outputs[i] for i, f in enumerate(autodiff_obj.backward_input_fields)})
            compiled_operation.backward(**{f.name: t for f, t in self.saved.items()})
            return tuple(self.saved[f] for f in autodiff_obj.backward_output_fields)

        cls = type(str(uuid.uuid4()), (torch.autograd.Function,), {})
        cls.saved = None
        cls.forward = forward
        cls.backward = backward
        cls.code = compiled_operation.code
        return cls()
    else:
        op = pystencils_autodiff.backends._pytorch.create_autograd_function(autodiff_obj,
                                                                            inputfield_to_tensor_dict,
                                                                            forward_loop,
                                                                            backward_loop,
                                                                            convert_tensors_to_arrays=False)
        return op
