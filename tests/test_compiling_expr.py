from __future__ import annotations

import pytest
from qadence2_expressions.operators import Expression, NativeDrive, X

from qadence2 import Register
from qadence2.compiler import code_compile


@pytest.mark.parametrize(
    "expr",
    [
        X(),
        NativeDrive(1.0, 1.0, 0.0, 0.0)(),
    ],
)
def test_expr_no_qubits(expr: Expression) -> None:
    with pytest.raises(ValueError):
        code_compile(expr, "fresnel1")


def test_expr_no_qubits_with_register() -> None:
    expr = X()
    assert code_compile(
        expr=expr,
        backend_name="pyqtorch",
        register=Register(grid_scale=1.0, grid_type="triangular", qubit_positions=[(0, 0)]),
    )
