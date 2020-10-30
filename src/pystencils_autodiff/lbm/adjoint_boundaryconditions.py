import pystencils
import pystencils_autodiff
from lbmpy.boundaries.boundaryconditions import Boundary
from lbmpy.boundaries.boundaryhandling import BoundaryOffsetInfo


class AdjointBoundaryCondition(Boundary):
    """ Creates adjoint LBM boundary condition from forward boundary condition """

    def __init__(self, forward_boundary_condition, time_constant_fields=[], constant_fields=[]):
        super().__init__("Adjoint" + forward_boundary_condition.name)
        self._forward_condition = forward_boundary_condition
        self._constant_fields = constant_fields
        self._time_constant_fields = time_constant_fields

    def __call__(self, pdf_field: pystencils_autodiff.AdjointField, direction_symbol, *args, **kwargs):

        # apply heuristics
        if pdf_field.name.startswith('diff'):
            forward_field = pystencils.Field.new_field_with_different_name(pdf_field, pdf_field.name[len('diff'):])
            pdf_field = pystencils_autodiff.AdjointField(forward_field)

        assert isinstance(pdf_field, pystencils_autodiff.AdjointField), \
            f'{pdf_field} should be a pystencils_autodiff.AdjointField to use AdjointBoundaryCondition'

        forward_field = pdf_field.corresponding_forward_field
        forward_assignments = self._forward_condition(forward_field, direction_symbol, *args, **kwargs)

        backward_assignments = pystencils_autodiff.create_backward_assignments(
            forward_assignments,
            diff_fields_prefix=pdf_field.name_prefix,
            time_constant_fields=self._time_constant_fields,
            constant_fields=self._constant_fields)
        assert backward_assignments.all_assignments, ("Must have a at least one read field in forward boundary "
                                                      "to have an meaningful adjoint boundary condition")

        return backward_assignments

    def __hash__(self):
        # All boundaries of these class behave equal -> should also be equal (as long as name is equal)
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, AdjointBoundaryCondition):
            return False
        return self._forward_condition == other._forward_condition


class AdjointNoSlip(Boundary):
    """ Bug-safe implementation of AdjointBoundaryCondition for NoSlip.
    Should be also safe to use AdjointBoundaryCondition(NoSlip()) """

    def __init__(self, name=None):
        """Set an optional name here, to mark boundaries, for example for force evaluations"""
        super().__init__(name)

    """No-Slip, (half-way) simple bounce back boundary condition, enforcing zero velocity at obstacle"""

    def __call__(self, pdf_field, direction_symbol, lb_method, **kwargs):
        neighbor = BoundaryOffsetInfo.offset_from_dir(
            direction_symbol, lb_method.dim)
        inverse_dir = BoundaryOffsetInfo.inv_dir(direction_symbol)
        return [pystencils.Assignment(pdf_field(direction_symbol), pdf_field[neighbor](inverse_dir))]

    def __hash__(self):
        # All boundaries of these class behave equal -> should also be equal (as long as name is equal)
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, AdjointNoSlip):
            return False
        return self.name == other.name
