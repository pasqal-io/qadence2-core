from __future__ import annotations

import numpy as np
from qadence2_expressions import RX, RY, add_qpu_directives, compile_to_model, parameter
from qadence2_platforms.compiler import compile_to_backend
from qadence2_platforms.abstracts import OnEnum


a = parameter("a")
expr = RX(1.57 * a)(0) * RY(0.707 * a**2)(0)
print(f"expression: {str(expr)}")

add_qpu_directives({"digital": True})
model = compile_to_model(expr)
print(f"model: {model}\n")

f_params = {"a": np.array([1.0])}
compiled_model = compile_to_backend(model, "fresnel1")
# res = compiled_model.sample(values=f_params, shots=10_000, on="emulator")
res = compiled_model.sample(values=f_params, shots=10_000, on=OnEnum.EMULATOR)



print(f"sample result: {res}")
