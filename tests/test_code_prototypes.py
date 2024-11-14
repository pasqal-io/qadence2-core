from __future__ import annotations

from typing import Iterable, Any

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from qadence2_expressions import (
    NativeDrive, Z, parameter, variable, Expression, function, promote,
    X, exp
)

from qadence2 import Register, code_compile
from qadence2.extensions.legacy import RX, QuantumModel, kron, RY, chain, CNOT, add, N


def test_pyq_basic_diff_v1() -> None:
    print("BASIC DIFF V1")
    x = parameter("x")
    expr = RX(0, x)
    model = QuantumModel(expr, backend="pyqtorch")

    x = torch.arange(0, 2 * torch.pi, 0.1, requires_grad=True)

    values = {"x": x}

    fx = model.expectation(values, Z(0)).squeeze()

    dfdx = torch.autograd.grad(
        outputs=fx,
        inputs=x,
        grad_outputs=torch.ones_like(x),
    )[0]

    plt.plot(x.detach(), fx.detach(), label="f(x)")
    plt.plot(x.detach(), dfdx.detach(), label="df/dx")
    plt.legend()
    plt.show()


def test_pyq_basic_training_v1() -> None:
    x = parameter("x")
    w = variable("w")

    expr = RX(0, w * x)

    model = QuantumModel(expr, "pyqtorch")

    w_target = 1.2

    def y_ground_truth(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.cos(x * w).to(dtype=torch.float64)

    n_train = 20

    # Training data
    x_train = torch.linspace(0, 2 * torch.pi, steps=n_train)
    y_train = y_ground_truth(x_train, w=w_target)

    n_test = 100

    # Test data
    x_test = torch.linspace(0, 2 * torch.pi, steps=n_test)
    y_pred_initial = model.expectation({"x": x_test}, Z(0))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_epochs = 500

    def loss_fn(x_train: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        out = model.expectation({"x": x_train}, Z(0))
        loss = criterion(out.squeeze(), y_train)
        return loss

    for i in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(x_train, y_train)
        loss.backward()
        optimizer.step()

    y_pred_final = model.expectation({"x": x_test}, Z(0)).squeeze()

    plt.plot(x_test.detach(), y_pred_final.detach(), label="Final prediction")
    plt.plot(x_test.detach(), y_pred_initial.detach(), label="Initial prediction")
    plt.title("Found w = " + str(float(model.vparams["w"][0])))
    plt.scatter(x_train, y_train, label="Training points")
    plt.legend()
    plt.show()


def test_pyq_basic_training_v2() -> None:
    # Function to fit:
    def f(_x: torch.Tensor) -> torch.Tensor:
        return _x**5

    xmin = -1.0
    xmax = 1.0
    n_test = 100

    x_test = torch.linspace(xmin, xmax, steps=n_test)
    y_test = f(x_test)

    def acos(_x: Any) -> Expression:
        return function("acos", promote(_x))

    n_qubits = 4

    x = parameter("x")

    fm = kron(RX(i, (i + 1) * acos(x)) for i in range(n_qubits))

    rots1 = kron(RX(i, f"theta_{i}") for i in range(n_qubits))
    rots2 = kron(RY(i, f"theta_{n_qubits + i}") for i in range(n_qubits))
    rots3 = kron(RX(i, f"theta_{2*n_qubits + i}") for i in range(n_qubits))

    cnots = chain(CNOT(i, i + 1) for i in range(n_qubits - 1))

    ansatz = rots1 * rots2 * rots3 * cnots * rots2

    expr = fm * ansatz

    obs = add(Z(i) for i in range(n_qubits))

    model = QuantumModel(expr, backend="pyqtorch")

    # Chebyshev FM does not accept x = -1, 1
    xmin = -0.99
    xmax = 0.99
    n_train = 20

    x_train = torch.linspace(xmin, xmax, steps=n_train)
    y_train = f(x_train)

    # Initial model prediction
    y_pred_initial = model.expectation({"x": x_test}, obs).detach()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    n_epochs = 500

    def loss_fn(x_train: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        out = model.expectation({"x": x_train}, obs)
        loss = criterion(out.squeeze(), y_train)
        return loss

    for i in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(x_train, y_train)
        loss.backward()
        optimizer.step()

    y_pred_final = model.expectation({"x": x_test}, obs).detach()

    plt.plot(x_test, y_pred_initial, label="Initial prediction")
    plt.plot(x_test, y_pred_final, label="Final prediction")
    plt.scatter(x_train, y_train, label="Training points")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.show()


@pytest.mark.xfail
def test_pyq_basic_rx_v1() -> None:
    n_qubits = 2

    # I will just copy the value from C6_DICT[60] since there's not such a dict in qadence2
    C6 = 865723.02

    omega = torch.tensor(torch.pi)
    delta = torch.tensor(0.0)
    phase = torch.tensor(0.0)
    duration = (parameter("theta") / omega) * 1000.0

    # With these values, we only get a term in X
    h_x = (omega / 2) * torch.cos(phase) * add(X(i) for i in range(n_qubits))

    # We will vary the angle of rotation between 0 and 2pi
    theta_vals = torch.arange(0, 2 * torch.pi, 0.01)

    # Now we can show the effect of the interaction term by changing the distance between the qubits
    distance = 8.0
    h_int = (C6 / distance**6) * (N(0) @ N(1))
    evolution = exp((h_x + h_int) * (duration / 1000.0))
    rotation_close = (
        code_compile(evolution, "pyqtorch")
        .expectation(observable=Z(0), values={"theta": theta_vals})
        .squeeze()
        .detach()
    )

    distance = 20.0
    h_int = (C6 / distance**6) * (N(0) @ N(1))
    evolution = exp((h_x + h_int) * (duration / 1000.0))
    rotation_far = (
        code_compile(evolution, "pyqtorch")
        .expectation(observable=Z(0), values={"theta": theta_vals})
        .squeeze()
        .detach()
    )

    plt.plot(theta_vals, torch.cos(theta_vals), linewidth=3, linestyle="dotted", label="Perfect RX")
    plt.plot(theta_vals, rotation_close, label="Drive on close atoms")
    plt.plot(theta_vals, rotation_far, label="Drive on far atoms")
    plt.legend()
    plt.show()


def test_fresnel1_basic_rx_v1() -> None:
    qubit_positions = [(0, 0), (0, 1)]

    # atoms close
    spacing = 1.0
    register = Register(
        grid_type="square",
        grid_scale=spacing,
        qubit_positions=qubit_positions
    )
    expr = NativeDrive(0.6, 0.5, 0.0, 0.0)()

    module = code_compile(expr, "fresnel1", register=register)
    rotation_close = -1. * module.expectation(observable=Z(0))[0]

    # atoms far away
    spacing = 80.0
    register = Register(
        grid_type="square",
        grid_scale=spacing,
        qubit_positions=qubit_positions
    )
    expr = NativeDrive(.6, 1.0, 0.0, 0.0)()

    module = code_compile(expr, "fresnel1", register=register)
    rotation_far = -1. * module.expectation(observable=Z(0))[0]

    # plot
    theta_vals = np.linspace(0, 2 * np.pi, len(rotation_close))

    plt.plot(theta_vals, np.cos(theta_vals), linewidth=3, linestyle="dotted",
             label="Perfect RX"
             )
    plt.plot(theta_vals, rotation_close, label="Drive on close atoms")
    plt.plot(theta_vals, rotation_far, label="Drive on far atoms")
    plt.legend()
    plt.show()
