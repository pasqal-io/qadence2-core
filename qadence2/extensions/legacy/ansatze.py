from __future__ import annotations

from typing import Any, Callable

from qadence2_expressions.operators import RX, RY, NOT, Expression

from qadence2.extensions.legacy.utils import ParadigmStrategy


def hea(strategy: ParadigmStrategy) -> Any:
    hea_fn = hea_strategy(strategy)


def hea_digital() -> Any:
    pass


def _rotation_digital(operators: list[Callable, ...] | None = None) -> Any:
    """
    operators: should be a list of operators, such as `[RX, RY, RX]`

    returns an Expression for digital operators
    """

    pass


def _entangler_digital() -> Any:
    pass


def hea_strategy(strategy: ParadigmStrategy) -> Any:
    hea_strategy = {
        ParadigmStrategy.DIGITAL: hea_digital,
    }
    hea_fn = hea_strategy.get(strategy)
    if hea_fn:
        return hea_fn
    raise ValueError(f"strategy {strategy} not supported.")
