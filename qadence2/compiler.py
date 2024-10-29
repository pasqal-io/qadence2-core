from __future__ import annotations

from typing import Any

from qadence2_expressions import compile_to_model
from qadence2_platforms import AbstractInterface
from qadence2_platforms.compiler import compile_to_backend


def code_compile(expr: Any, backend_name: str, **settings: Any) -> AbstractInterface:
    model = compile_to_model(expr)
    print(f"{model=}\n{model.inputs=}")
    return compile_to_backend(model, backend_name)
