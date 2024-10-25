from __future__ import annotations

from qadence2_expressions.core.expression import Expression
from qadence2_expressions import X, Y, RX, parameter, compile_to_model
from qadence2_ir.types import Model
from qadence2_platforms.compiler import compile_to_backend

from qadence2.extensions.legacy.model import QuantumModel
from qadence2.compiler import code_compile


# TODO: generate random valid expressions with hypothesis

def test_expressions_ir() -> None:
    a = parameter("a")
    expr = Y(0) * RX(a)(0)
    model = compile_to_model(expr)
    backend = compile_to_backend(model, "pyqtorch")

