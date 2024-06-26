from __future__ import annotations

from typing import Any

from qadence_core.std.legacy.quantum_circuit import QuantumCircuit


class QuantumModel:
    def __init__(
        self, circuit: QuantumCircuit, observable: Any | None = None, **_: Any
    ):
        self.circuit = circuit
