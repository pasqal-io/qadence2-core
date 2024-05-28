from __future__ import annotations
from typing import Any, Iterable, Union, Protocol, TypeVar

from qadence_core.platforms.model import Register as DeviceRegister


Elem = TypeVar("Elem")
Shape = TypeVar("Shape")
Number = TypeVar("Number")


class GraphLike(Protocol[Elem]):
    def __iter__(self) -> Iterable:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Elem:
        ...


class AdjacencyMatrixLike(Protocol[Elem, Shape, Number]):
    def __iter__(self) -> Iterable:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Elem:
        ...

    @property
    def shape(self) -> Shape:
        ...

    @property
    def real(self) -> Number:
        ...

    @property
    def imag(self) -> Number:
        ...


class Register:
    def __init__(self, data: Union[GraphLike, AdjacencyMatrixLike]):
        self.data = data

    def convert(self) -> DeviceRegister:
        pass
