import tensorflow as tf
from tensorflow.compat.v1 import get_default_graph, py_func

import pystencils_autodiff

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
        return tf.py_func(helper_backward,
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
