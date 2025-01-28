from __future__ import annotations

import math
from enum import Enum, auto
from typing import Generator

from qadence2_expressions.core.expression import Expression

PI = math.pi


class ParadigmStrategy(Enum):
    DIGITAL = auto()
    ANALOG = auto()
    SDAQC = auto()
    BDAQC = auto()
    RYDBERG = auto()


def add(*expr: Generator | list[Expression]) -> Expression:
    if len(expr) > 0 and isinstance(expr[0], Generator):
        return Expression.add(*tuple(*expr))
    return Expression.add(*expr)


def mul(*expr: Generator | list[Expression]) -> Expression:
    if len(expr) > 0 and isinstance(expr[0], Generator):
        return Expression.mul(*tuple(*expr))
    return Expression.mul(*expr)


def chain(*expr: Generator | list[Expression]) -> Expression:
    return mul(*expr)


def kron(*expr: Generator | list[Expression]) -> Expression:
    if len(expr) > 0 and isinstance(expr[0], Generator):
        return Expression.kron(*tuple(*expr))
    return Expression.kron(*expr)


def pow(*expr: Generator | list[Expression]) -> Expression:
    if len(expr) > 0 and isinstance(expr[0], Generator):
        return Expression.pow(*tuple(*expr))
    return Expression.pow(*expr)
