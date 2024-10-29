from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from qadence2_expressions import RX, Z, parameter, variable
from qadence2_platforms.backends.pyqtorch.interface import Interface

from qadence2.extensions.legacy import QuantumModel

if __name__ == "__main__":
    print("Qadence 2 code prototype for PYQTorch with basic training v1\n\n")

    x = parameter("x")
    w = variable("w")

    expr = RX(w * x)(0)

    model: Interface = QuantumModel(expr, "pyqtorch")

    w_target = 1.2

    def y_ground_truth(x, w):
        return torch.cos(x * w).to(dtype=torch.float64)

    n_train = 20

    # Training data
    x_train = torch.linspace(0, 2 * torch.pi, steps=n_train)
    y_train = y_ground_truth(x_train, w=w_target)

    n_test = 100

    # Test data
    x_test = torch.linspace(0, 2 * torch.pi, steps=n_test)
    y_pred_initial = model.expectation(values={"x": x_test}, observable=Z(0))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_epochs = 500

    def loss_fn(x_train, y_train):
        out = model.expectation(values={"x": x_train}, observable=Z(0))
        loss = criterion(out.squeeze(), y_train)
        return loss

    for i in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(x_train, y_train)
        loss.backward()
        optimizer.step()

    y_pred_final = model.expectation(values={"x": x_test}, observable=Z(0)).squeeze()

    plt.plot(x_test.detach(), y_pred_final.detach(), label="Final prediction")
    plt.plot(x_test.detach(), y_pred_initial.detach(), label="Initial prediction")
    plt.title("Found w = " + str(float(model.vparams["w"][0])))
    plt.scatter(x_train, y_train, label="Training points")
    plt.legend()
    plt.show()
