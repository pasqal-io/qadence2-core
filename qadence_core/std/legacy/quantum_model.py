from __future__ import annotations

from qadence_core.std.legacy.quantum_circuit import QuantumCircuit


class QuantumModel:
    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
