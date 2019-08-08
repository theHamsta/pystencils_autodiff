.. image:: https://badge.fury.io/py/pystencils-autodiff.svg
   :target: https://badge.fury.io/py/pystencils-autodiff
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/pystencils-autodiff/badge/?version=latest
    :target: https://pystencils-autodiff.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status===================

.. image:: https://travis-ci.org/theHamsta/pystencils_autodiff.svg?branch=master
    :target: https://travis-ci.org/theHamsta/pystencils_autodiff

.. image:: https://codecov.io/gh/theHamsta/pystencils_autodiff/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/theHamsta/pystencils_autodiff

pystencils_autodiff
===================

This repo adds automatic differentiation to `pystencils <https://i10git.cs.fau.de/seitz/pystencils>`_.

Installation
------------

Install via pip:

.. code-block:: bash

   pip install pystencils-autodiff

or if you downloaded this `repository <https://github.com/theHamsta/pystencils_autodiff>`_ using:

.. code-block:: bash

   pip install -e .

Then, you can access the submodule `pystencils.autodiff`.

.. code-block:: python

    import pystencils.autodiff

Usage
-----

Create a `pystencils.AssignmentCollection` with pystencils:

.. code-block:: python

    import sympy
    import pystencils

    z, x, y = pystencils.fields("z, y, x: [20,30]")

    forward_assignments = pystencils.AssignmentCollection({
        z[0, 0]: x[0, 0] * sympy.log(x[0, 0] * y[0, 0])
    })

    print(forward_assignments)


.. code-block:: python

    Subexpressions:
    Main Assignments:
         z[0,0] ← y_C*log(x_C*y_C)
   
You can then obtain the corresponding backward assignments:

.. code-block:: python

    from pystencils.autodiff import AutoDiffOp, create_backward_assignments
    backward_assignments = create_backward_assignments(forward_assignments)

    print(backward_assignments)

You can see the derivatives with respective to the two inputs multiplied by the gradient `diffz_C` of the output `z_C`.

.. code-block:: python

    Subexpressions:
    Main Assignments:
        \hat{y}[0,0] ← diffz_C*(log(x_C*y_C) + 1)
        \hat{x}[0,0] ← diffz_C*y_C/x_C

You can also use the class `AutoDiffOp` to obtain both the assignments (if you are curious) and auto-differentiable operations for Tensorflow...

.. code-block:: python

    op = AutoDiffOp(forward_assignments)
    backward_assignments = op.backward_assignments   

    x_tensor = pystencils.autodiff.tf_variable_from_field(x)
    y_tensor = pystencils.autodiff.tf_variable_from_field(y)
    tensorflow_op = op.create_tensorflow_op({x: x_tensor, y: y_tensor}, backend='tensorflow')

... or Torch:

.. code-block:: python

    x_tensor = pystencils.autodiff.torch_tensor_from_field(x, cuda=False, requires_grad=True)
    y_tensor = pystencils.autodiff.torch_tensor_from_field(y, cuda=False, requires_grad=True)

    z_tensor = op.create_tensorflow_op({x: x_tensor, y: y_tensor}, backend='torch')
