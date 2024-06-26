from __future__ import annotations
from typing import Any, Callable, Union
from functools import singledispatchmethod

import torch
import numpy as np
import networkx as nx

from qadence_core.expressions.expr import Expr
from qadence_core.types.register import Register, GraphLike, AdjacencyMatrixLike


class QuantumCircuit:
    @singledispatchmethod
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
