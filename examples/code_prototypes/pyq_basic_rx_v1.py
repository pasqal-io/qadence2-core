from __future__ import annotations

import torch
import matplotlib.pyplot as plt

from qadence2_expressions import parameter, exp

from qadence2.compiler import code_compile
from qadence2.extensions.legacy import N, X, Z, add

n_qubits = 2

# I will just copy the value from C6_DICT[60] since there's not such a dict in qadence2
C6 = 865723.02

omega = torch.tensor(torch.pi)
delta = torch.tensor(0.0)
phase = torch.tensor(0.0)
duration = (parameter("theta") / omega) * 1000.

# With these values, we only get a term in X
h_x = (omega / 2) * torch.cos(phase) * add(X(i) for i in range(n_qubits))

# We will vary the angle of rotation between 0 and 2pi
theta_vals = torch.arange(0, 2 * torch.pi, 0.01)

# Now we can show the effect of the interaction term by changing the distance between the qubits
distance = 8.0
h_int = (C6 / distance**6) * (N(0) @ N(1))
evolution = exp((h_x + h_int) * (duration / 1000.))
print(f"{type((h_x + h_int) * (duration / 1000.))} {((h_x + h_int) * (duration / 1000.)).max_index} {(h_x + h_int) * (duration / 1000.)}")
rotation_close = code_compile(evolution, "pyqtorch").expectation(observable = Z(0), values = {"theta": theta_vals}).squeeze().detach()

distance = 20.0
h_int = (C6 / distance**6) * (N(0) @ N(1))
evolution = exp((h_x + h_int) * (duration / 1000.))
rotation_far = code_compile(evolution, "pyqtorch").expectation(observable = Z(0), values = {"theta": theta_vals}).squeeze().detach()

plt.plot(theta_vals, torch.cos(theta_vals), linewidth=3, linestyle = "dotted", label = "Perfect RX")
plt.plot(theta_vals, rotation_close, label = "Drive on close atoms")
plt.plot(theta_vals, rotation_far, label = "Drive on far atoms")
plt.legend()
plt.show()
