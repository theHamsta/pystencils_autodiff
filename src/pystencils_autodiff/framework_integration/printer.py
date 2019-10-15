import pystencils.backends.cbackend
from pystencils.kernelparameters import FieldPointerSymbol


class FrameworkIntegrationPrinter(pystencils.backends.cbackend.CBackend):
    """
    Printer for C++ modules.

    Its prints wrapper code as an extension of :class:`pystencils.backends.cbackend.CBackend`.
    Only kernel code is printed with the backend it was generated for.

    """

    def __init__(self):
        super().__init__(sympy_printer=None,
                         dialect='c')

    def _print_WrapperFunction(self, node):
        super_result = super()._print_KernelFunction(node)
        return super_result.replace('FUNC_PREFIX ', '')

    def _print_KernelFunction(self, node):
        if node.backend == 'gpucuda':
            prefix = '#define FUNC_PREFIX __global__\n'
            kernel_code = pystencils.backends.cbackend.generate_c(node, dialect='cuda', with_globals=False)
        else:
            prefix = '#define FUNC_PREFIX\n'
            kernel_code = pystencils.backends.cbackend.generate_c(node, dialect='c', with_globals=False)
        return prefix + kernel_code

    def _print_KernelFunctionCall(self, node):

        function = node.kernel_function
        parameters = function.get_parameters()
        function_name = function.function_name
        indent = " " * (len(function_name) + 1)
        function_arguments = (",\n" + indent) .join("%s" % (s.symbol.name) for s in parameters)
        if function.backend == "gpucuda":
            written_fields = function.fields_written
            shape = list(written_fields)[0].shape
            # assert all(shape == f.shape for f in written_fields)

            block_and_thread_numbers = function.indexing.call_parameters(shape)
            launch_blocks = 'dim3{' + ', '.join(self.sympy_printer.doprint(i)
                                                for i in block_and_thread_numbers['block']) + '}'
            launch_grid = 'dim3{' + ', '.join(self.sympy_printer.doprint(i)
                                              for i in block_and_thread_numbers['grid']) + '}'
            return f"{function_name}<<< {launch_blocks}, {launch_grid} >>>(\n{indent}{function_arguments});"
        else:
            return f"{function_name}({function_arguments});"
        return node.__repr__()

    def _print_JinjaCppFile(self, node):
        return str(node)

    def _print_DestructuringBindingsForFieldClass(self, node):
        # Define all undefined symbols
        undefined_field_symbols = node.symbols_defined
        destructuring_bindings = ["%s %s = %s.%s;" %
                                  (u.dtype,
                                   u.name,
                                   u.field_name if hasattr(u, 'field_name') else u.field_names[0],
                                   node.CLASS_TO_MEMBER_DICT[u.__class__].format(
                                       dtype=(u.dtype.base_type if type(u) == FieldPointerSymbol else ""),
                                       field_name=(u.field_name if hasattr(u, "field_name") else ""),
                                       dim=("" if type(u) == FieldPointerSymbol else u.coordinate)
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
