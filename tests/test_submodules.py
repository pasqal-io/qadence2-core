from qadence_core.platforms.model import Parameter, Instruction
from qadence_core.expressions.expr import Expr


def test_model_calls():
    assert Parameter("theta", 1, mutable=False)
    assert Parameter("omega", 1, mutable=True)
    assert Instruction("not", (0,))
    assert Instruction("rx", (1,), Parameter("theta1", 1, mutable=False))


def test_expr_call():
    assert Expr("a")
    assert Expr("a") + Expr("b")
