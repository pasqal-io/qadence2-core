from __future__ import annotations

from typing import Any

from qadence2_platforms import AbstractInterface

from qadence2.compiler import code_compile


def QuantumModel(expr: Any, backend: str, **settings: Any) -> AbstractInterface:
    return code_compile(expr, backend, **settings)
