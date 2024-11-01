from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

from qadence2_expressions import (
    compile_to_model,
    add_qpu_directives,
    add_settings,
    add_grid_options,
    reset_ir_options,
    set_grid_scale,
    set_grid_type,
    set_number_qubits,
    set_qubits_positions
)
from qadence2_platforms import AbstractInterface
from qadence2_platforms.compiler import compile_to_backend


def code_compile(
    expr: Any,
    backend_name: str,
    register: Register | dict | None = None,
    directives: Directives | dict | None = None,
    settings: Settings | dict | None = None,
    **kwargs: Any
) -> AbstractInterface:
    set_config_compiler(register, directives, settings)
    model = compile_to_model(expr)
    return compile_to_backend(model, backend_name)


@dataclass
class Register:
    grid_type: Literal["linear", "square", "triangular"]
    grid_scale: float = field(default=1.0)
    qubit_positions: list = field(default_factory=list)
    number_qubits: Union[int, None] = field(default=None)
    grid_options: dict = field(default_factory=dict)

    def add_configs(self) -> None:
        set_grid_scale(self.grid_scale)
        set_grid_type(self.grid_type)
        if self.qubit_positions:
            set_qubits_positions(self.qubit_positions)
        else:
            set_number_qubits(self.number_qubits)

        if self.grid_options:
            add_grid_options(self.grid_options)


class Directives(dict):
    def add_configs(self) -> None:
        add_qpu_directives(self)


class Settings(dict):
    def add_configs(self) -> None:
        add_qpu_directives(self)


def set_config_register(register: Register | dict) -> None:
    """
    It is assumed to have all the values to pass to the IR register.

    Args:
        register: a Register class or a dict containing register data such as
            grid type, grid scale, qubit positions, and additional configs.
    """
    if not isinstance(register, Register):
        register = Register(**register)
    register.add_configs()


def set_config_directives(directives: Directives | dict) -> None:
    if not isinstance(directives, Directives):
        directives = Directives(**directives)
    directives.add_configs()


def set_config_settings(settings: Settings | dict) -> None:
    if not isinstance(settings, Settings):
        settings = Settings(**settings)
    settings.add_configs()


def set_config_compiler(
    register: Register | dict | None = None,
    directives: Directives | dict | None = None,
    settings: Settings | dict | None = None
) -> None:
    if register:
        set_config_register(register)
    if directives:
        set_config_directives(directives)
    if settings:
        set_config_settings(settings)
