import sympy as sp

import pystencils
from pystencils.data_types import TypedSymbol, create_type
from pystencils_autodiff.framework_integration.astnodes import DynamicFunction
from pystencils_autodiff.framework_integration.printer import FrameworkIntegrationPrinter
from pystencils_autodiff.framework_integration.types import CustomCppType, TemplateType


def test_dynamic_function():
    x, y = pystencils.fields('x, y:  float32[3d]')

    a = sp.symbols('a')

    my_fun_call = DynamicFunction(TypedSymbol('my_fun',
                                              'std::function<double(double)>'), create_type('double'), a)

    assignments = pystencils.AssignmentCollection({
        y.center: x.center + my_fun_call
    })

    ast = pystencils.create_kernel(assignments)
    pystencils.show_code(ast, custom_backend=FrameworkIntegrationPrinter())

    template_fun_call = DynamicFunction(TypedSymbol('my_fun',
                                                    TemplateType('Functor_T')), create_type('double'), a, x.center)

    assignments = pystencils.AssignmentCollection({
        y.center: x.center + template_fun_call
    })

    ast = pystencils.create_kernel(assignments)
    pystencils.show_code(ast, custom_backend=FrameworkIntegrationPrinter())


def test_dynamic_matrix():
    x, y = pystencils.fields('x, y:  float32[3d]')
    try:
        from pystencils.data_types import TypedMatrixSymbol
    except ImportError:
        import pytest
        pytest.skip()

    a = sp.symbols('a')

    A = TypedMatrixSymbol('A', 3, 1, create_type('double'), 'Vector3<double>')

    my_fun_call = DynamicFunction(TypedSymbol('my_fun',
                                              'std::function<Vector3<double>(double)>'), A.dtype, a)

    assignments = pystencils.AssignmentCollection({
        A:  my_fun_call,
        y.center: A[0] + A[1] + A[2]
    })

    ast = pystencils.create_kernel(assignments)
    pystencils.show_code(ast, custom_backend=FrameworkIntegrationPrinter())


def test_typed_matrix():
    try:
        from pystencils.data_types import TypedMatrixSymbol
    except ImportError:
        import pytest
        pytest.skip()

    x, y = pystencils.fields('x, y:  float32[3d]')

    A = TypedMatrixSymbol('A', 3, 1, create_type('double'), CustomCppType('Vector3<real_t>'))

    assignments = pystencils.AssignmentCollection({
        y.center: A[0] + A[1] + A[2]
    })

    ast = pystencils.create_kernel(assignments)
    pystencils.show_code(ast, custom_backend=FrameworkIntegrationPrinter())


def test_dynamic_matrix_location_dependent():
    try:
        from pystencils.data_types import TypedMatrixSymbol
    except ImportError:
        import pytest
        pytest.skip()

    x, y = pystencils.fields('x, y:  float32[3d]')

    A = TypedMatrixSymbol('A', 3, 1, create_type('double'), CustomCppType('Vector3<double>'))

    my_fun_call = DynamicFunction(TypedSymbol('my_fun',
                                              'std::function<Vector3<double>(int, int, int)>'),
                                  A.dtype,
                                  *pystencils.x_vector(3))

    assignments = pystencils.AssignmentCollection({
        A:  my_fun_call,
        y.center: A[0] + A[1] + A[2]
    })

    ast = pystencils.create_kernel(assignments)
    pystencils.show_code(ast, custom_backend=FrameworkIntegrationPrinter())

    my_fun_call = DynamicFunction(TypedSymbol('my_fun',
                                              TemplateType('Functor_T')), A.dtype, *pystencils.x_vector(3))

    assignments = pystencils.AssignmentCollection({
        A:  my_fun_call,
        y.center: A[0] + A[1] + A[2]
    })

    ast = pystencils.create_kernel(assignments)
    pystencils.show_code(ast, custom_backend=FrameworkIntegrationPrinter())
