import uuid
from collections import OrderedDict

from pystencils_autodiff.backends._pytorch import numpy_dtype_to_torch
from pystencils_autodiff.backends.astnodes import TorchModule

try:
    import torch
except ImportError:
    pass


def create_autograd_function(autodiff_obj, inputfield_to_tensor_dict):
    field_to_tensor_dict = inputfield_to_tensor_dict

    # Allocate output tensor for forward and backward pass
    for field in autodiff_obj.forward_output_fields + autodiff_obj.backward_output_fields:
        field_to_tensor_dict[field] = torch.zeros(
            *field.shape,
            dtype=numpy_dtype_to_torch(field.dtype.numpy_dtype),
            device=list(inputfield_to_tensor_dict.values())[0].device)

    all_tensors = field_to_tensor_dict.values()
    is_cuda = all(a.is_cuda for a in all_tensors)

    if is_cuda:
        forward_ast = autodiff_obj.forward_ast_gpu
        backward_ast = autodiff_obj.backward_ast_gpu
    else:
        forward_ast = autodiff_obj.forward_ast_cpu
        backward_ast = autodiff_obj.backward_ast_cpu

    op_name = autodiff_obj.op_name + str(uuid.uuid4())
    compiled_op = TorchModule(op_name, [forward_ast, backward_ast])

    output_tensors = OrderedDict({f.name: field_to_tensor_dict[f] for f in autodiff_obj.forward_output_fields})
    backward_output_tensors = OrderedDict(
        {f.name: field_to_tensor_dict[f] for f in autodiff_obj.backward_output_fields})

    def forward(self, **input_tensors):

        self.save_for_backward(**input_tensors)

        getattr(compiled_op, "call_" + forward_ast.function_name)(**input_tensors, **output_tensors)

        return output_tensors.values()

    def backward(self, *grad_outputs):
        gradients = {f.name: grad_outputs[i] for i, f in enumerate(autodiff_obj.backward_input_fields)}
        saved = self.saved_tensors

        getattr(compiled_op, "call_" + backward_ast.function_name)(**gradients, **saved, **backward_output_tensors)

        return backward_output_tensors.values()

    cls = type(op_name, (torch.autograd.Function,), {})
    cls.forward = forward
    cls.backward = backward
    return cls
