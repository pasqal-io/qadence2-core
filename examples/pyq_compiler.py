from __future__ import annotations

import pyqtorch as pyq
import torch
from qadence2_expressions import RX, RY, add_qpu_directives, compile_to_model, parameter
from qadence2_platforms.compiler import compile_to_backend


if __name__ == "__main__":
    a = parameter("a")
    # expr = RX(1.57 * a)(0) * RY(0.707 * a**2)(0)
    expr = RX(a)(0)
    print(f"expression: {str(expr)}")

    add_qpu_directives({"digital": True})
    model = compile_to_model(expr)
    print(f"model: {model}\n")

    f_params = {"a": torch.tensor(1.0, requires_grad=True)}
    compiled_model = compile_to_backend(model, "pyqtorch")
    res = compiled_model.sample(values=f_params, shots=10_000)
    print(f"sample result: {res}")

    wf = compiled_model.run(state=pyq.zero_state(2), values=f_params)
    dfdx = torch.autograd.grad(wf, f_params["a"], torch.ones_like(wf))[0]
    print(f"{dfdx = }\n")
