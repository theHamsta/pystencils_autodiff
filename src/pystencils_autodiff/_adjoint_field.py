import pystencils
from pystencils.astnodes import FieldShapeSymbol, FieldStrideSymbol

r"""Determines how adjoint fields will be denoted in LaTeX output in terms of the forward field representation %s
Default: r'\hat{%s}' """
ADJOINT_FIELD_LATEX_HIGHLIGHT = r"\hat{%s}"


class AdjointField(pystencils.Field):
    """Field representing adjoint variables to a Field representing the forward variables"""

    def __init__(self, forward_field, name_prefix='diff'):
        new_name = name_prefix + forward_field.name
        super().__init__(new_name, pystencils.FieldType.GENERIC
                         if forward_field.field_type != pystencils.FieldType.BUFFER
                         else pystencils.FieldType.BUFFER, forward_field._dtype,
                         forward_field._layout, forward_field.shape, forward_field.strides)
        self.corresponding_forward_field = forward_field
        self.name_prefix = name_prefix

        # Eliminate references to forward fields that might not be present in backward kernels
        self.shape = tuple(FieldShapeSymbol([self.name], s.coordinate) if isinstance(
            s, FieldShapeSymbol) else s for s in self.shape)
        self.strides = tuple(FieldStrideSymbol(self.name, s.coordinate) if isinstance(
            s, FieldStrideSymbol) else s for s in self.strides)

        if forward_field.latex_name:
            self.latex_name = ADJOINT_FIELD_LATEX_HIGHLIGHT % forward_field.latex_name
        else:
            self.latex_name = ADJOINT_FIELD_LATEX_HIGHLIGHT % forward_field.name
