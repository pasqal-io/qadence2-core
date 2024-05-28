from __future__ import annotations

from qadence_core.expressions.expr import Expr
from qadence_core.types.operators import H, RZ, CNOT
from qadence_core.expressions.transform.evaluate import prod
from qadence_core.std.data_conversion import convert_data


def rzz(ctrl, tgt, theta) -> Expr:
    return CNOT(ctrl, tgt) * RZ(theta)(tgt) * CNOT(ctrl, tgt)


def simple_featuremap(theta, pairs) -> Expr:
    block = prod(rzz(ctrl, tgt, theta) for ctrl, tgt in pairs)
    return H() * block * H()


def featuremap_embedding(data) -> Expr:
    return simple_featuremap()


def ising_embedding() -> Expr:
    pass
