from __future__ import annotations

from qadence2_expressions import RX, Y, compile_to_model, parameter
from qadence2_ir.types import Alloc, Model

# TODO: generate random valid expressions with hypothesis


def test_expressions_ir() -> None:
    a = parameter("a")
    expr = Y(0) * RX(a)(0)
    model: Model = compile_to_model(expr)
    assert len(model.inputs) == 1
    assert model.inputs.get("a") == Alloc(size=1, trainable=False)
    assert model.register.num_qubits == 1
    assert model.register.grid_scale == 1.0
    assert model.directives == dict()
    assert len(model.instructions) == 2
    # TODO: continue the testing
    # backend = compile_to_backend(model, "pyqtorch")
