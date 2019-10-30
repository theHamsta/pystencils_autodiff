from collections import OrderedDict
from itertools import chain

from pystencils_autodiff.backends._pytorch import numpy_dtype_to_torch
from pystencils_autodiff.backends.astnodes import TorchModule
from pystencils_autodiff.tensorflow_jit import _hash


def create_autograd_function(autodiff_obj, use_cuda):
    import torch
    field_to_tensor_dict = dict()
    # Allocate output tensor for forward and backward pass
    # for field in autodiff_obj.forward_output_fields + autodiff_obj.backward_output_fields:
    # field_to_tensor_dict[field] = torch.zeros(
    # *field.shape,
    # dtype=numpy_dtype_to_torch(field.dtype.numpy_dtype),
    # device=list(inputfield_to_tensor_dict.values())[0].device)

    if use_cuda:
        forward_ast = autodiff_obj.forward_ast_gpu
        backward_ast = autodiff_obj.backward_ast_gpu
    else:
        forward_ast = autodiff_obj.forward_ast_cpu
        backward_ast = autodiff_obj.backward_ast_cpu

    op_name = f'{autodiff_obj.op_name}_{_hash(str(autodiff_obj).encode()).hexdigest()}'
    module = TorchModule(op_name, [forward_ast, backward_ast])
    compiled_op = module.compile()

    # print(TorchModule(op_name, [forward_ast, backward_ast]))
    # wrapper = module.atoms(WrapperFunction)
    # forward_wrapper_ast = [w for w in wrapper if w.function_name == "call_" + forward_ast.function_name][0]
    # backward_wrapper_ast = [w for w in wrapper if w.function_name == "call_" + backward_ast.function_name][0]
    # forward_parameters = [str(p.symbol) for p in forward_wrapper_ast.get_parameters()]
    # backward_parameters = [str(p.symbol) for p in backward_wrapper_ast.get_parameters()]
    class_kwargs = dict()

    def forward(self, *args, **kwargs):

        kwargs.update(class_kwargs)
        # TODO: drop contiguous requirement
        if use_cuda:
            args = [a.contiguous().cuda() if isinstance(a, torch.Tensor) else a for a in args]
            kwargs = {k: v.contiguous().cuda() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        else:
            args = [a.contiguous().cpu() if isinstance(a, torch.Tensor) else a for a in args]
            kwargs = {k: v.contiguous().cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        # assert all(f.shape == args[i].shape for i, f in enumerate(autodiff_obj.forward_input_fields)
        # if not any(isinstance(s, sp.Symbol) for s in args[i].shape))
        # assert all(f.strides == tuple(args[i].stride(j) for j in range(args[i].ndim))
        # for i, f in enumerate(autodiff_obj.forward_input_fields))
        # for field in autodiff_obj.forward_output_fields:
        # field_to_tensor_dict[field] = torch.zeros(
        # field.shape,
        # dtype=numpy_dtype_to_torch(field.dtype.numpy_dtype),
        # device=args[0].device)

        kwargs.update({f.name: args[i] for i, f in enumerate(
            autodiff_obj.forward_input_fields) if f in forward_ast.fields_accessed if i < len(args)})

        for field in autodiff_obj.forward_output_fields:
            if field.name not in kwargs:
                kwargs[field.name] = torch.zeros(
                    field.shape,
                    dtype=numpy_dtype_to_torch(field.dtype.numpy_dtype),
                    device=next(chain(args, kwargs.values())).device)
        output_tensors = OrderedDict({f.name:
                                      field_to_tensor_dict.get(f, kwargs[f.name])
                                      for f in autodiff_obj.forward_output_fields})
        field_to_tensor_dict.update(kwargs)
        kwargs.update(output_tensors)

        self.saved_for_backward = kwargs

        getattr(compiled_op, "call_" + forward_ast.function_name)(**kwargs)

        return tuple(output_tensors.values())

    def backward(self, *grad_outputs):
        if use_cuda:
            grad_outputs = [a.contiguous().cuda() for a in grad_outputs]
        else:
            grad_outputs = [a.contiguous().cpu() for a in grad_outputs]
        gradients = {f.name: grad_outputs[i] for i, f in enumerate(autodiff_obj.backward_input_fields)}
        assert all(f.shape == grad_outputs[i].shape for i, f in enumerate(autodiff_obj.backward_input_fields))
        assert all(f.strides == tuple(grad_outputs[i].stride(j) for j in range(grad_outputs[i].ndim))
                   for i, f in enumerate(autodiff_obj.backward_input_fields))
        assert all(a.is_cuda == use_cuda for a in grad_outputs), "Some of the tensors where on the wrong device. "
        f"Op was compiled for CUDA: {str(use_cuda)}"

        for field in autodiff_obj.backward_output_fields:
            backward_output_tensors = OrderedDict({f.name: torch.zeros(field.shape,
                                                                       dtype=numpy_dtype_to_torch(
                                                                           field.dtype.numpy_dtype),
                                                                       device=grad_outputs[0].device)
                                                   for f in autodiff_obj.backward_output_fields})
        field_names = [f.name for f in backward_ast.fields_accessed]
        kwargs = {**gradients, **self.saved_for_backward, **backward_output_tensors}
        kwargs = {k: v for k, v in kwargs.items() if k in field_names}
        getattr(compiled_op, "call_" + backward_ast.function_name)(**kwargs)

        return tuple(backward_output_tensors.values())

    cls = type(op_name, (torch.autograd.Function,), {})
    cls.class_kwargs = class_kwargs
    cls.forward = forward
    cls.backward = backward
    cls.kernel = forward
    cls.ast = module
    cls.parameters = forward_ast.get_parameters()
    cls.forward_ast = forward_ast
    cls.backward_ast = backward_ast
    cls.num_regs = None
    cls.code = str(module)

    return cls
