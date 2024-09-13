import numpy as np
from qadence2_platforms.compiler import compile_to_backend
from qadence2_expressions import *


a = parameter("a")
expr = RX(1.57 * a)(0) * RY(0.707 * a ** 2)(0)
print(f"expression: {str(expr)}")

add_qpu_directives({"digital": True})
model = compile_to_model(expr)
print(f"model: {model}\n")

f_params = {"a": np.array([1.0])}
compiled_model = compile_to_backend("fresnel1", model)
res = compiled_model.sample(values=f_params, shots=10_000, on="emulator")
print(f"sample result: {res}")
