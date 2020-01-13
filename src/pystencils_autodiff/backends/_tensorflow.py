from collections.abc import Iterable

import stringcase
from tensorflow.compat.v1 import get_default_graph, py_func

import pystencils_autodiff
from pystencils_autodiff.backends.astnodes import TensorflowModule
from pystencils_autodiff.tensorflow_jit import _hash

_num_generated_ops = 0


def _py_func(func, inp, Tout, stateful=False, name=None, grad=None):
    """
    Copied from random internet forum. It seems to be important to give
    PyFunc to give an random name in override map to properly register gradients

    PyFunc defined as given by Tensorflow
    :param func: Custom Function
    :param inp: Function Inputs
    :param Tout: Output Type of out Custom Function
    :param stateful: Calculate Gradients when stateful is True
    :param name: Name of the PyFunction
    :param grad: Custom Gradient Function
    :return:
    """
    import tensorflow as tf
    # Generate Random Gradient name in order to avoid conflicts with inbuilt names
    global _num_generated_ops
    rnd_name = 'PyFuncGrad' + str(_num_generated_ops) + 'ABC@a1b2c3'
    _num_generated_ops += 1

    # Register Tensorflow Gradient
    tf.RegisterGradient(rnd_name)(grad)

    # Get current graph
    g = get_default_graph()

    # Add gradient override map
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return py_func(func, inp, Tout, stateful=stateful, name=name)


def native_tensorflowop_from_autodiffop(autodiff_obj: pystencils_autodiff.AutoDiffOp,
                                        use_cuda):

    import tensorflow as tf
    if use_cuda:
        forward_ast = autodiff_obj.forward_ast_gpu
        backward_ast = autodiff_obj.backward_ast_gpu
    else:
        forward_ast = autodiff_obj.forward_ast_cpu
        backward_ast = autodiff_obj.backward_ast_cpu

    op_name = f'{autodiff_obj.op_name.lower()}_hash{_hash(str((autodiff_obj, autodiff_obj.forward_input_fields, autodiff_obj.backward_input_fields)).encode()).hexdigest()}'  # noqa
    if use_cuda:
        op_name += '_cuda'
    forward_ast.function_name = op_name + "_forward"
    backward_ast.function_name = op_name + "_backward"
    module = TensorflowModule(op_name, [forward_ast, backward_ast])
    compiled_op = module.compile()

    backward_func = getattr(compiled_op, stringcase.snakecase(
        stringcase.pascalcase("call_" + backward_ast.function_name)))
    grad_fields = [f for f in autodiff_obj.backward_input_fields if f not in autodiff_obj.forward_input_fields]

    def gradient_calculation(op, *grad):
        if not isinstance(grad, Iterable):
            grad = [grad]

        return backward_func(**{grad_fields[i].name: g for i, g in enumerate(grad)},
                             **{autodiff_obj.forward_input_fields[i].name: inp for i, inp in enumerate(op.inputs)
                                if autodiff_obj.forward_input_fields[i] in backward_ast.fields_accessed})

    try:
        tf.RegisterGradient(stringcase.pascalcase("call_" + forward_ast.function_name))(
            gradient_calculation
        )
    except Exception:
        pass

    return getattr(compiled_op, stringcase.snakecase(stringcase.pascalcase("call_" + forward_ast.function_name)))


def tensorflowop_from_autodiffop(autodiffop: pystencils_autodiff.AutoDiffOp,
                                 inputfield_tensor_dict,
                                 forward_function,
                                 backward_function):

    def helper_forward(*args):
        kwargs = dict()
        for i in range(len(args)):
            if args[i] is not None:
                kwargs[autodiffop.forward_input_fields[i].name] = args[i]

        rtn_dict = forward_function(**kwargs)
        return [rtn_dict[o.name] for o in autodiffop._forward_output_fields]

    def helper_backward(*args):
        kwargs = dict()
        for i in range(len(args)):
            if i < len(autodiffop.forward_input_fields):
                kwargs[autodiffop.forward_input_fields[i].name] = args[i]
            else:
                kwargs[autodiffop._backward_input_fields[i -
                                                         len(autodiffop.forward_input_fields)].name] = args[i]
        rtn_dict = backward_function(**kwargs)
        return [rtn_dict[o.name] for o in autodiffop._backward_output_fields]

    def backward(op, *grad):
        return py_func(helper_backward,
                       [*op.inputs,
                           *grad],
                       [f.dtype.numpy_dtype for f in autodiffop._backward_output_fields],
                       name=autodiffop.op_name + '_backward',
                       stateful=False)

    output_tensors = _py_func(helper_forward,
                              [inputfield_tensor_dict[f]
                               for f in autodiffop.forward_input_fields],
                              [f.dtype.numpy_dtype for f in autodiffop._forward_output_fields],
                              name=autodiffop.op_name, stateful=False, grad=backward)

    return output_tensors
