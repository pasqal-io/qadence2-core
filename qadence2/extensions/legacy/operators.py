from __future__ import annotations

import qadence2_expressions.operators as ops
from qadence2_expressions.core.constructors import promote, unitary_hermitian_operator, parametric_operator
from qadence2_expressions.operators import _join_rotation
from qadence2_expressions import variable
from qadence2_expressions.core.expression import Expression

from .utils import PI

# The N = (1/2)(I-Z) operator
N = ops.Z1


def CNOT(control: int, target: int) -> Expression:
    return ops.NOT(target=(target,), control=(control,))


def RX(target: int, parameters: Expression | str | float) -> Expression:
    return ops.RX(angle=_get_variable(parameters))(target)


def RY(target: int, parameters: Expression | str | float) -> Expression:
    return ops.RY(angle=_get_variable(parameters))(target)


def RZ(target: int, parameters: Expression | str | float) -> Expression:
    return ops.RZ(angle=_get_variable(parameters))(target)


def _get_variable(expr: Expression | str | float) -> Expression:
    if isinstance(expr, str):
        return variable(expr)
    if isinstance(expr, (Expression, float)):
        return expr


def T(target: int) -> Expression:
    return unitary_hermitian_operator("T")(target=(target,))


def PHASE(target: int, parameters: Expression | str | float) -> Expression:
    return parametric_operator("PHASE", promote(_get_variable(parameters)), join=_join_rotation)(target=(target,))


def CPHASE(control: int, target: int, parameters: Expression | str | float) -> Expression:
    return parametric_operator("CPHASE", promote(_get_variable(parameters)), join=_join_rotation)(target=(target,), control=(control,))

