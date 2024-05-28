from __future__ import annotations
from functools import singledispatch

# TODO: transform it into different submodules to avoid installing/loading unused packages?

import torch
import numpy as np
import networkx as nx


"""
Data conversion used to transform various data into expressions.

It is heavily user-input dependent and should be considered before establishing a pattern
for conversion.
"""


@singledispatch
def convert_data(data):
    pass


@convert_data.register
def _(data: torch.Tensor):
    pass


@convert_data.register
def _(data: np.ndarray):
    pass


@convert_data.register
def _(data: nx.Graph):
    pass


@convert_data.register
def _(data: list):
    pass
