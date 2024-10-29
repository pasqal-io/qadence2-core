from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from qadence2_expressions import RX, Z, parameter

from qadence2.extensions.legacy import QuantumModel

if __name__ == "__main__":

    print("Qadence 2 code prototype for PyQTorch with basic differentiation v1\n\n")

    x = parameter("x")
    expr = RX(x)(0)
    model = QuantumModel(expr, backend="pyqtorch")

    x = torch.arange(0, 2 * torch.pi, 0.1, requires_grad=True)

    values = {"x": x}

    fx = model.expectation(values=values, observable=Z(0)).squeeze()

    dfdx = torch.autograd.grad(
        outputs=fx,
        inputs=x,
        grad_outputs=torch.ones_like(x),
    )[0]

    plt.plot(x.detach(), fx.detach(), label="f(x)")
    plt.plot(x.detach(), dfdx.detach(), label="df/dx")
    plt.legend()
    plt.show()
