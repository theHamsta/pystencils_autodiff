from collections import OrderedDict

from pystencils_autodiff.backends._pytorch import numpy_dtype_to_torch
from pystencils_autodiff.backends.astnodes import TorchModule
from pystencils_autodiff.tensorflow_jit import _hash

try:
    import torch
except ImportError:
    pass


def create_autograd_function(autodiff_obj, use_cuda):
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

    def forward(self, *args):

        if use_cuda:
            args = [a.contiguous().cuda() for a in args]
        else:
            args = [a.contiguous().cpu() for a in args]

        input_tensors = dict()
        input_tensors.update({f.name: args[i] for i, f in enumerate(
            autodiff_obj.forward_input_fields) if f in forward_ast.fields_accessed})
        assert all(f.shape == args[i].shape for i, f in enumerate(autodiff_obj.forward_input_fields))
        assert all(f.strides == tuple(args[i].stride(j) for j in range(args[i].ndim))
                   for i, f in enumerate(autodiff_obj.forward_input_fields))
        for field in autodiff_obj.forward_output_fields:
            field_to_tensor_dict[field] = torch.zeros(
                field.shape,
                dtype=numpy_dtype_to_torch(field.dtype.numpy_dtype),
                device=args[0].device)
        output_tensors = OrderedDict({f.name: field_to_tensor_dict[f] for f in autodiff_obj.forward_output_fields})

        self.save_for_backward(*args)

        getattr(compiled_op, "call_" + forward_ast.function_name)(**input_tensors, **output_tensors)

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
        assert all(a.is_cuda == use_cuda for a in grad_outputs), f"Some of the tensors where on the wrong device. " \
            f"Op was compiled for CUDA: {str(use_cuda)}"
        saved = {f.name: self.saved_tensors[i] for i, f in enumerate(
            autodiff_obj.forward_input_fields) if f in backward_ast.fields_accessed}
        for field in autodiff_obj.backward_output_fields:
            field_to_tensor_dict[field] = torch.zeros(
                field.shape,
                dtype=numpy_dtype_to_torch(field.dtype.numpy_dtype),
                device=grad_outputs[0].device)

        backward_output_tensors = OrderedDict(
            {f.name: field_to_tensor_dict[f] for f in autodiff_obj.backward_output_fields})
        getattr(compiled_op, "call_" + backward_ast.function_name)(**gradients, **saved, **backward_output_tensors)

        return tuple(backward_output_tensors.values())

    cls = type(op_name, (torch.autograd.Function,), {})
    cls.forward = forward
    cls.backward = backward
    return cls
