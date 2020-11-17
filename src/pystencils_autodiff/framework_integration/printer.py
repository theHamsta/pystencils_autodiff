import functools

import sympy as sp

import pystencils.backends.cbackend
from pystencils.astnodes import KernelFunction
from pystencils.data_types import TypedSymbol
from pystencils.kernelparameters import FieldPointerSymbol
from pystencils_autodiff.framework_integration.types import TemplateType


class FrameworkIntegrationPrinter(pystencils.backends.cbackend.CBackend):
    """
    Printer for C++ modules.

    Its prints wrapper code as an extension of :class:`pystencils.backends.cbackend.CBackend`.
    Only kernel code is printed with the backend it was generated for.

    """

    def __init__(self):
        super().__init__(dialect='c')
        self.sympy_printer.__class__._print_DynamicFunction = self._print_DynamicFunction
        self.sympy_printer.__class__._print_MeshNormalFunctor = self._print_DynamicFunction
        self.sympy_printer.__class__._print_MatrixElement = self._print_MatrixElement
        self.sympy_printer.__class__._print_TypedMatrixElement = self._print_MatrixElement

    def _print(self, node):
        from pystencils_autodiff.framework_integration.astnodes import JinjaCppFile
        if isinstance(node, JinjaCppFile):
            node.printer = self
        if isinstance(node, sp.Expr):
            return self.sympy_printer._print(node)
        else:
            return super()._print(node)

    def _print_BlockWithoutBraces(self, node):
        block_contents = "\n".join([self._print(child) for child in node.args])
        return "\n%s\n" % (''.join(block_contents.splitlines(True)))

    def _print_WrapperFunction(self, node):
        if node.return_value:
            node._body._nodes.append(node.return_value)
        super_result = self._print_KernelFunction_extended(node)
        if self._signatureOnly:
            super_result += ';'
        return super_result.replace('FUNC_PREFIX ', '')

    def _print_TextureDeclaration(self, node):
        return str(node)

    def _print_KernelFunction_extended(self, node: KernelFunction):
        return_type = node.return_type if hasattr(node, 'return_type') and node.return_type else 'void'
        function_arguments = [f"{self._print(s.symbol.dtype)} {s.symbol.name}"
                              for s in node.get_parameters() if hasattr(s.symbol, 'dtype')]
        launch_bounds = ""
        if self._dialect == 'cuda':
            max_threads = node.indexing.max_threads_per_block()
            if max_threads:
                launch_bounds = f"__launch_bounds__({max_threads}) "
        indent = ',\n' + ' ' * len(f'FUNC_PREFIX {launch_bounds} {self._print(return_type)} {node.function_name}')
        func_declaration = f"FUNC_PREFIX {launch_bounds} {self._print(return_type)} {node.function_name}({indent.join(function_arguments)})"  # noqa
        if self._signatureOnly:
            return func_declaration

        body = self._print(node.body)
        return func_declaration + "\n" + body

    def _print_KernelFunction(self, node, return_type='void'):
        if node.backend == 'gpucuda':
            prefix = '#define FUNC_PREFIX static __global__\n'
            kernel_code = pystencils.backends.cbackend.generate_c(node, dialect='cuda', with_globals=False)
        else:
            prefix = '#define FUNC_PREFIX static\n'
            kernel_code = pystencils.backends.cbackend.generate_c(node, dialect='c', with_globals=False)
        template_types = sorted([x.dtype for x in node.atoms(TypedSymbol)
                                 if isinstance(x.dtype, TemplateType)], key=str)
        template_types = list(map(lambda x: 'class ' + str(x), template_types))
        if template_types:
            prefix = f'{prefix}template <{",".join(template_types)}>\n'
        if self._signatureOnly:
            suffix = ';'
        else:
            suffix = ''

        return prefix + kernel_code + suffix

    def _print_FunctionCall(self, node):

        function = node.kernel_function
        parameters = function.get_parameters()
        function_name = function.function_name
        indent = " " * (len(function_name) + 1)
        function_arguments = (",\n" + indent) .join("%s" % (s.symbol.name) for s in parameters)
        if function.backend == "gpucuda":
            written_fields = function.fields_written
            shape = list(written_fields)[0].spatial_shape
            # assert all(shape == f.shape for f in written_fields)

            # TODO(seitz): this is not correct for indexed kernels!
            block_and_thread_numbers = function.indexing.call_parameters(shape)
            launch_blocks = 'dim3{' + ', '.join(self.sympy_printer.doprint(i)
                                                for i in block_and_thread_numbers['block']) + '}'
            launch_grid = 'dim3{' + ', '.join(self.sympy_printer.doprint(i)
                                              for i in block_and_thread_numbers['grid']) + '}'
            return f"{function_name} <<< {launch_grid}, {launch_blocks} >>>(\n{indent}{function_arguments});"
        else:
            return f"{function_name}({function_arguments});"
        return node.__repr__()

    def _print_JinjaCppFile(self, node):
        return str(node)

    def _print_DestructuringBindingsForFieldClass(self, node):
        # Define all undefined symbols
        undefined_field_symbols = node.symbols_defined
        fields_dtype = {u.field_name:
                        u.dtype.base_type for u in undefined_field_symbols if isinstance(u, FieldPointerSymbol)}
        destructuring_bindings = ["%s %s = %s.%s;" %
                                  (u.dtype,
                                   u.name,
                                   (u.field_name if hasattr(u, 'field_name') else u.field_names[0])
                                   + (node.field_suffix if hasattr(node, 'field_suffix') else ''),
                                   node.CLASS_TO_MEMBER_DICT[u.__class__].format(
                                       dtype=(u.dtype.base_type if type(u) == FieldPointerSymbol
                                              else ((fields_dtype.get(u.field_name
                                                                      if hasattr(u, 'field_name')
                                                                      else u.field_names[0]))
                                                    or (fields_dtype.get('diff' + u.field_name
                                                                         if hasattr(u, 'field_name')
                                                                         else 'diff' + u.field_names[0])))),
                                       field_name=(u.field_name if hasattr(u, "field_name") else ""),
                                       dim=("" if type(u) == FieldPointerSymbol else u.coordinate),
                                       dim_letter=("" if type(u) == FieldPointerSymbol else 'xyz'[u.coordinate])
                                   )
                                   )
                                  for u in undefined_field_symbols
                                  ]

        destructuring_bindings.sort()  # only for code aesthetics
        return "{\n" + self._indent + \
            ("\n" + self._indent).join(destructuring_bindings) + \
            "\n" + self._indent + \
            ("\n" + self._indent).join(self._print(node.body).splitlines()) + \
            "\n}"

    def _print_Timeloop(self, node):
        children_string = '\n   '.join(self._print(c) for c in node.children)
        return f"""for( {node.loop_symbol.dtype} {node.loop_symbol}={node.loop_start};  {node.loop_symbol}<= {node.loop_end} ; {node.loop_symbol} += {node.loop_increment} ) {{
    {children_string}
}}"""  # noqa

    def _print_SwapBuffer(self, node):
        return f"""std::swap({node.first_array}, {node.second_array});"""

    def _print_DynamicFunction(self, expr):
        name = expr.name
        arg_str = ', '.join(self._print(a) for a in expr.args[2:])
        return f'{name}({arg_str})'

    def _print_MatrixElement(self, expr):
        name = expr.name
        if expr.args[0].args[1] == 1 or expr.args[0].args[2] == 1 or (hasattr(expr.args[0], 'linear_indexing')
                                                                      and expr.args[0].linear_indexing):
            return f'{name}[{self._print(expr.args[1])}]'
        else:
            arg_str = ', '.join(self._print(a) for a in expr.args[1:])
            return f'{name}({arg_str})'

    def _print_CustomCodeNode(self, node):
        super_code = super()._print_CustomCodeNode(node)
        if super_code:
            # Without leading new line
            return super_code[1:]
        else:
            return super_code


class DebugFrameworkPrinter(FrameworkIntegrationPrinter):
    """
    Printer with information on nodes inlined in code as comments.

    Should not be used in production, will modify your SymPy printer, destroy your whole life!
    """

    def __init__(self):
        super().__init__()
        self.sympy_printer._old_print = self.sympy_printer._print
        self.sympy_printer.__class__._print = self._print

    def _print(self, node):
        if isinstance(node, sp.Expr):
            return self.sympy_printer._old_print(node) + f'/* {node.__class__.__name__}: free_symbols: {node.free_symbols} */'  # noqa
        elif isinstance(node, pystencils.astnodes.Node):
            return super()._print(node) + f'/* {node.__class__.__name__} symbols_undefined: {node.undefined_symbols}, symbols_defined: {node.symbols_defined}, args {[a if isinstance(a,str) else a.__class__.__name__ for a in node.args]} */'  # noqa
        else:
            return super()._print(node)


show_code = functools.partial(pystencils.show_code, custom_backend=FrameworkIntegrationPrinter())
get_code_str = functools.partial(pystencils.get_code_str, custom_backend=FrameworkIntegrationPrinter())
