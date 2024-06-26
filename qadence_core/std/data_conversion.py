"""
Data conversion used to transform various data into expressions.

It is heavily user-input dependent and should be considered before establishing a pattern
for conversion.
"""

from __future__ import annotations

from functools import singledispatch

# TODO: transform it into different submodules to avoid installing/loading unused packages?
import networkx as nx
import numpy as np
import torch

from qadence_core.expressions.expr import Expr


@singledispatch
def convert_data(data) -> Expr:
    raise NotImplementedError()


@convert_data.register
def _(data: torch.Tensor) -> Expr:
    raise NotImplementedError()


@convert_data.register
def _(data: np.ndarray) -> Expr:
    raise NotImplementedError()


@convert_data.register
def _(data: nx.Graph) -> Expr:
    raise NotImplementedError()


@convert_data.register
def _(data: list) -> Expr:
    raise NotImplementedError()
