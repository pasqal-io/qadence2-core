from __future__ import annotations

from qadence_core.expressions.expr import Symbol


class Parameter(Symbol):
    def __init__(self, name: str, trainable: bool):
        self.trainable = trainable
        super().__init__(name)


class Constant(Parameter):
    def __init__(self, name: str):
        super().__init__(name, False)


class Variable(Parameter):
    def __init__(self, name: str):
        super().__init__(name, True)

    def __str__(self) -> str:
        return f"${self.name}"
