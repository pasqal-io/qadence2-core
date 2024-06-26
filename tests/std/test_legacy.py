from __future__ import annotations

from qadence_core.expressions.expr import QSymbol
from qadence_core.std.legacy import QuantumCircuit, QuantumModel

Z = QSymbol("Z")


def test_legacy_quantum_circuit() -> None:
    assert QuantumCircuit(Z(0) * Z(1))


def test_legay_quantum_model() -> None:
    circuit = QuantumCircuit(Z(0) * Z(1))
    assert QuantumModel(circuit)
