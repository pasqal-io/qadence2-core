from __future__ import annotations

from matplotlib import pyplot as plt
import torch

from qadence2_expressions import Z, parameter, function, promote

from qadence2.extensions.legacy import QuantumModel, RX, RY, CNOT, add, chain, kron


if __name__ == "__main__":

    # Function to fit:
    def f(_x):
        return _x**5

    xmin = -1.0
    xmax = 1.0
    n_test = 100

    x_test = torch.linspace(xmin, xmax, steps = n_test)
    y_test = f(x_test)


    def acos(_x):
        return function("acos", promote(_x))


    n_qubits = 4

    x = parameter("x")


    fm = kron(RX(i, (i+1) * acos(x)) for i in range(n_qubits))

    rots1 = kron(RX(i, f"theta_{i}") for i in range(n_qubits))
    rots2 = kron(RY(i, f"theta_{n_qubits + i}") for i in range(n_qubits))
    rots3 = kron(RX(i, f"theta_{2*n_qubits + i}") for i in range(n_qubits))

    cnots = chain(CNOT(i, i+1) for i in range(n_qubits-1))

    ansatz = rots1 * rots2 * rots3 * cnots * rots2

    expr = fm * ansatz

    obs = add(Z(i) for i in range(n_qubits))

    model = QuantumModel(expr, backend="pyqtorch")

    # Chebyshev FM does not accept x = -1, 1
    xmin = -0.99
    xmax = 0.99
    n_train = 20

    x_train = torch.linspace(xmin, xmax, steps = n_train)
    y_train = f(x_train)

    # Initial model prediction
    y_pred_initial = model.expectation({"x": x_test}, obs).detach()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

    n_epochs = 500


    def loss_fn(x_train, y_train):
        out = model.expectation({"x": x_train}, obs)
        loss = criterion(out.squeeze(), y_train)
        return loss


    for i in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(x_train, y_train)
        loss.backward()
        optimizer.step()


    y_pred_final = model.expectation({"x": x_test}, obs).detach()

    plt.plot(x_test, y_pred_initial, label = "Initial prediction")
    plt.plot(x_test, y_pred_final, label = "Final prediction")
    plt.scatter(x_train, y_train, label = "Training points")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.show()
