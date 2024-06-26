from __future__ import annotations

from typing import Any, Union

from qadence_core.expressions.expr import Expr
from qadence_core.expressions.transform.evaluate import prod
from qadence_core.types.operators import CNOT, RZ, H
from qadence_core.types.parameter import Parameter

Scalar = float
NumericType = Union[Parameter, Scalar]
SupportType = tuple[int, ...]


def rzz(ctrl: int, tgt: int, theta: NumericType) -> Any:
    return CNOT(ctrl, tgt) * RZ(theta)(tgt) * CNOT(ctrl, tgt)


def simple_featuremap(
    theta: NumericType, pairs: tuple[SupportType, SupportType]
) -> Any:
    block = prod(rzz(ctrl, tgt, theta) for ctrl, tgt in pairs)
    return H() * block * H()


def featuremap_embedding(data: Any) -> Expr:
    # TODO: finish it
    # return simple_featuremap()
    raise NotImplementedError()


def ising_embedding() -> Expr:
    # TODO: finish it
    raise NotImplementedError()
