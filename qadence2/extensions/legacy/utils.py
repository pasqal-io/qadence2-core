from __future__ import annotations

from enum import Enum, auto
from typing import Generator

from qadence2_expressions.core.expression import Expression


class ParadigmStrategy(Enum):
    DIGITAL = auto()
    ANALOG = auto()
    SDAQC = auto()
    BDAQC = auto()
    RYDBERG = auto()


def add(expr: Generator | list[Expression]) -> Expression:
    return Expression.add(*tuple(expr))


def mul(expr: Generator | list[Expression]) -> Expression:
    return Expression.mul(*tuple(expr))


def chain(expr: Generator | list[Expression]) -> Expression:
    return mul(expr)


def kron(expr: Generator | list[Expression]) -> Expression:
    return Expression.kron(*tuple(expr))


def pow(expr: Generator | list[Expression]) -> Expression:
    return Expression.pow(*tuple(expr))
