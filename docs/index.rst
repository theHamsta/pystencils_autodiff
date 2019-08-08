===================
pystencils-autodiff
===================

This is the documentation of **pystencils-autodiff**.

This document assumes that you are already familiar with `pystencils <https://i10git.cs.fau.de/seitz/pystencils>`_.
If not, here is a good `tutorial to start <http://pycodegen.pages.walberla.net/pystencils/notebooks/01_tutorial_getting_started.html>`_.

Installation of this Auto-Diff Extension
----------------------------------------

Install via pip :

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

Create a :class:`pystencils.AssignmentCollection` with pystencils:

.. testcode::

    import sympy
    import pystencils

    z, x, y = pystencils.fields("z, y, x: [20,30]")

    forward_assignments = pystencils.AssignmentCollection({
        z[0, 0]: x[0, 0] * sympy.log(x[0, 0] * y[0, 0])
    })

    print(forward_assignments)


.. testoutput::
    :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

    Subexpressions:
    Main Assignments:
         z[0,0] ← y_C*log(x_C*y_C)
   
You can then obtain the corresponding backward assignments:

.. testcode::

    from pystencils.autodiff import AutoDiffOp, create_backward_assignments
    backward_assignments = create_backward_assignments(forward_assignments)

    # Sorting for reprudcible outputs
    backward_assignments.main_assignments = sorted(backward_assignments.main_assignments, key=lambda a: str(a))

    print(backward_assignments)

You can see the derivatives with respective to the two inputs multiplied by the gradient `diffz_C` of the output `z_C`.

.. testoutput::
    :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

    Subexpressions:
    Main Assignments:
        \hat{x}[0,0] ← diffz_C*y_C/x_C
        \hat{y}[0,0] ← diffz_C*(log(x_C*y_C) + 1)

You can also use the class :class:`.autodiff.AutoDiffOp` to obtain both the assignments (if you are curious) and auto-differentiable operations for Tensorflow...

.. testcode::

    op = AutoDiffOp(forward_assignments)
    backward_assignments = op.backward_assignments   

    x_tensor = pystencils.autodiff.tf_variable_from_field(x)
    y_tensor = pystencils.autodiff.tf_variable_from_field(y)
    tensorflow_op = op.create_tensorflow_op({x: x_tensor, y: y_tensor}, backend='tensorflow')

... or Torch:

.. testcode::

    x_tensor = pystencils.autodiff.torch_tensor_from_field(x, cuda=False, requires_grad=True)
    y_tensor = pystencils.autodiff.torch_tensor_from_field(y, cuda=False, requires_grad=True)

    z_tensor = op.create_tensorflow_op({x: x_tensor, y: y_tensor}, backend='torch')


Contents
========

.. toctree::
   :maxdepth: 2

   License <license>
   Authors <authors>
   Changelog <changelog>
   Module Reference <api/modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
