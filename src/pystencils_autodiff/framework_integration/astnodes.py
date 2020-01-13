# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""
Astnodes useful for the generations of C modules for frameworks (apart from waLBerla)

waLBerla currently uses `pystencils-walberla <https://pypi.org/project/pystencils-walberla/>`_.
"""
import itertools
from collections.abc import Iterable
from typing import Any, List, Set

import jinja2
import sympy as sp

import pystencils
from pystencils.astnodes import KernelFunction, Node, NodeOrExpr, ResolvedFieldAccess
from pystencils.backends.cbackend import CustomCodeNode
from pystencils.data_types import TypedSymbol
from pystencils.kernelparameters import FieldPointerSymbol, FieldShapeSymbol, FieldStrideSymbol
from pystencils_autodiff.framework_integration.printer import FrameworkIntegrationPrinter
from pystencils_autodiff.framework_integration.texture_astnodes import NativeTextureBinding


class DestructuringBindingsForFieldClass(Node):
    """
    Defines all variables needed for describing a field (shape, pointer, strides)
    """
    headers = ['<PyStencilsField.h>']
    CLASS_TO_MEMBER_DICT = {
        FieldPointerSymbol: "data<{dtype}>",
        FieldShapeSymbol: "shape[{dim}]",
        FieldStrideSymbol: "stride[{dim}]"
    }
    CLASS_NAME_TEMPLATE = "PyStencilsField<{dtype}, {ndim}>"
    ARGS_AS_REFERENCE = True

    @property
    def fields_accessed(self) -> Set['ResolvedFieldAccess']:
        """Set of Field instances: fields which are accessed inside this kernel function"""

        from pystencils.interpolation_astnodes import InterpolatorAccess
        return set(o.field for o in self.atoms(ResolvedFieldAccess) | self.atoms(InterpolatorAccess)) \
            | set(itertools.chain.from_iterable((k.kernel_function.fields_accessed
                                                 for k in self.atoms(FunctionCall))))

    def __init__(self, body):
        super(DestructuringBindingsForFieldClass, self).__init__()
        self.body = body
        body.parent = self

    @property
    def args(self) -> List[NodeOrExpr]:
        """Returns all arguments/children of this node."""
        return [self.body]

    @property
    def symbols_defined(self) -> Set[sp.Symbol]:
        """Set of symbols which are defined by this node."""
        undefined_field_symbols = {s for s in self.body.undefined_symbols
                                   if isinstance(s, (FieldPointerSymbol, FieldShapeSymbol, FieldStrideSymbol))}
        return undefined_field_symbols

    @property
    def undefined_symbols(self) -> Set[sp.Symbol]:
        field_map = {f.name: f for f in self.fields_accessed}
        undefined_field_symbols = self.symbols_defined
        corresponding_field_names = {s.field_name for s in undefined_field_symbols if hasattr(s, 'field_name')}
        corresponding_field_names |= {s.field_names[0] for s in undefined_field_symbols if hasattr(s, 'field_names')}
        return {TypedSymbol(f,
                            self.CLASS_NAME_TEMPLATE.format(dtype=field_map[f].dtype,
                                                            ndim=field_map[f].ndim) + ('&'
                                                                                       if self.ARGS_AS_REFERENCE
                                                                                       else ''))
                for f in corresponding_field_names} | (self.body.undefined_symbols - undefined_field_symbols)

    def subs(self, subs_dict) -> None:
        """Inplace! substitute, similar to sympy's but modifies the AST inplace."""
        self.body.subs(subs_dict)

    def __repr__(self):
        return f'Destructuring of Tensors {self.symbols_defined}\n' + self.body.__repr__()

    @property
    def func(self):
        return self.__class__

    def atoms(self, arg_type) -> Set[Any]:
        return self.body.atoms(arg_type) | {s for s in self.symbols_defined if isinstance(s, arg_type)}


class FunctionCall(Node):
    """
    AST nodes representing a call of a :class:`pystencils.astnodes.KernelFunction`
    """

    def __init__(self, kernel_function_node: pystencils.astnodes.KernelFunction):
        self.kernel_function = kernel_function_node

    @property
    def args(self):
        return [p.symbol for p in self.kernel_function.get_parameters()] + [self.kernel_function]

    @property
    def symbols_defined(self) -> Set[sp.Symbol]:
        return {}

    @property
    def undefined_symbols(self) -> Set[sp.Symbol]:
        return {p.symbol for p in self.kernel_function.get_parameters()}

    def subs(self, subs_dict) -> None:
        for a in self.args:
            a.subs(subs_dict)

    @property
    def func(self):
        return self.__class__

    def __repr__(self):
        return f"call {self.kernel_function.function_name}{self.kernel_function.get_parameters()}"


class WrapperFunction(pystencils.astnodes.KernelFunction):

    def __init__(self, body, function_name='wrapper', target='cpu', backend='c'):
        super().__init__(body, target, backend, compile_function=None, ghost_layers=0)
        self.function_name = function_name


def generate_kernel_call(kernel_function):
    try:
        from pystencils.interpolation_astnodes import TextureAccess
        from pystencils.kernelparameters import FieldPointerSymbol

        textures = {a.texture for a in kernel_function.atoms(TextureAccess)}
        texture_uploads = [NativeTextureBinding(t, FieldPointerSymbol(t.field.name, t.field.dtype, const=True))
                           for t in textures]
    except ImportError:
        texture_uploads = []

    # debug_print = CustomCodeNode(
    # 'std::cout << "hallo" << __PRETTY_FUNCTION__ << std::endl;\ngpuErrchk(cudaPeekAtLastError());' \
    # '\ncudaDeviceSynchronize();', set(), set())

    if texture_uploads:
        block = pystencils.astnodes.Block([
            CudaErrorCheck(),
            *texture_uploads,
            CudaErrorCheck(),
            FunctionCall(kernel_function),
            CudaErrorCheck(),
        ])
    elif kernel_function.backend == 'gpucuda':
        return pystencils.astnodes.Block([CudaErrorCheck(),
                                          FunctionCall(kernel_function),
                                          CudaErrorCheck()])
    else:
        return pystencils.astnodes.Block([FunctionCall(kernel_function)])

    return block


class JinjaCppFile(Node):
    TEMPLATE: jinja2.Template = None

    def __init__(self, ast_dict):
        self.ast_dict = ast_dict
        self.printer = FrameworkIntegrationPrinter()
        Node.__init__(self)

    @property
    def args(self):
        """Returns all arguments/children of this node."""
        ast_nodes = [a for a in self.ast_dict.values() if isinstance(a, (Node, str))]
        iterables_of_ast_nodes = [a for a in self.ast_dict.values() if isinstance(a, Iterable)]
        return ast_nodes + list(itertools.chain.from_iterable(iterables_of_ast_nodes))

    @property
    def symbols_defined(self):
        """Set of symbols which are defined by this node."""
        return set()

    @property
    def undefined_symbols(self):
        """Symbols which are used but are not defined inside this node."""
        return set()

    def _print(self, node):
        if isinstance(node, Node):
            return self.printer(node)
        else:
            return str(node)

    def atoms(self, arg_type) -> Set[Any]:
        """Returns a set of all descendants recursively, which are an instance of the given type."""
        result = set()
        for arg in self.args:
            if isinstance(arg, arg_type):
                result.add(arg)
            if hasattr(arg, 'atoms'):
                result.update(arg.atoms(arg_type))
        return result

    @property
    def is_cuda(self):
        return any(f.backend == 'gpucuda' for f in self.atoms(KernelFunction))

    def __str__(self):
        assert self.TEMPLATE, f"Template of {self.__class__} must be set"
        render_dict = {k: (self._print(v) if not isinstance(v, (pystencils.Field, pystencils.TypedSymbol)) else v)
                       if not isinstance(v, Iterable) or isinstance(v, str)
                       else [(self._print(a)
                              if not isinstance(a, (pystencils.Field, pystencils.TypedSymbol))
                              else a)
                             for a in v]
                       for k, v in self.ast_dict.items()}

        # TODO: possibly costly tree traversal
        render_dict.update({"headers": pystencils.backends.cbackend.get_headers(self)})
        render_dict.update({"globals":
                            [self.printer(g) for g in pystencils.backends.cbackend.get_global_declarations(self)]})

        return self.TEMPLATE.render(render_dict)

    def __repr__(self):
        return f'{str(self.__class__)}:\n {self.TEMPLATE.render(self.ast_dict)}'


class CudaErrorCheckDefinition(CustomCodeNode):
    def __init__(self):
        super().__init__(self.code, [], [])

    function_name = 'gpuErrchk'
    code = """
# ifdef __GNUC__
# define gpuErrchk(ans) { gpuAssert((ans), __PRETTY_FUNCTION__, __FILE__, __LINE__); }
inline static void gpuAssert(cudaError_t code, const char* function, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA error: %s: in %s %s:%d\\n", cudaGetErrorString(code), function, file, line);
      if (abort) exit(code);
   }
}
# else
# define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline static void gpuAssert(cudaError_t code, const char* function, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA error: %s: %s:%d\\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
# endif
"""
    headers = ['<cuda.h>']


class CudaErrorCheck(CustomCodeNode):
    """
    Checks whether the last call to the CUDA API was successful and panics in the negative case.

    .. code:: c++

        # define gpuErrchk(ans) { gpuAssert((ans), __PRETTY_FUNCTION__, __FILE__, __LINE__); }
        inline static void gpuAssert(cudaError_t code, const char* function, const char *file, int line, bool abort=true)
        {
           if (code != cudaSuccess)
           {
              fprintf(stderr,"CUDA error: %s: %s:%d\\n", cudaGetErrorString(code), file, line);
              if (abort) exit(code);
           }
        }

        ...

        gpuErrchk(cudaPeekAtLastError());

    """  # noqa

    def __init__(self):
        super().__init__(f'{self.err_check_function.function_name}(cudaPeekAtLastError());', [], [])

    err_check_function = CudaErrorCheckDefinition()
    required_global_declarations = [err_check_function]
    headers = ['<cuda.h>']
