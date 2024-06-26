from __future__ import annotations

from functools import singledispatchmethod
from typing import Any, Callable

import networkx as nx
import numpy as np
import torch

from qadence_core.expressions.expr import Expr


class QuantumCircuit:
    @singledispatchmethod  # type: ignore [misc]
    def __init__(self, data, **kwargs):
        raise NotImplementedError

    @__init__.register
    def _(self, data: Expr, **_: Any):
        self.circuit = data

    @__init__.register
    def _(self, data: np.ndarray, embedding: Callable):
        self.circuit = embedding(data)

    @__init__.register
    def _(self, data: torch.Tensor, embedding: Callable):
        self.circuit = embedding(data)

    @__init__.register
    def _(self, data: list, embedding: Callable):
        self.circuit = embedding(data)

    @__init__.register
    def _(self, data: nx.Graph, embedding: Callable):
        self.circuit = embedding(data)
