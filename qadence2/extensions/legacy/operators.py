from __future__ import annotations

from typing import Any

import qadence2_expressions.operators as ops
from qadence2_expressions import variable
from qadence2_expressions.core.expression import Expression


# The N = (1/2)(I-Z) operator
N = ops.Z1


def CNOT(target: int, control: int) -> Expression:
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
