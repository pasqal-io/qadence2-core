
# Transition Guide
This document is for Qadence users to use Qadence2 funcationalities with Qadence format.
For Qadence2 code details, please refer to https://github.com/pasqal-io/qadence2-core.

## Operators

Qadence operators can be used in the same way (including the order of target and control) in Qadence 2.

Qadence
```python exec="on" source="material-block" html="1" session="getting_started"
from qadence import RX, CNOT

rx = RX(0, 0.5)
cnot = CNOT(0, 1)
```

Qadence2
```python exec="on" source="material-block" html="1" session="getting_started"
from qadence2.extensions.legacy import RX, CNOT

rx = RX(0, 0.5)
cnot = CNOT(0, 1)
```

## Block System

Building quantum expressions with `chain` and `kron` is also identical.

Qadence
```python exec="on" source="material-block" html="1" session="getting_started"
from qadence import X, Y, chain, kron

chain_0 = chain(X(0), Y(0))
chain_1 = chain(X(1), Y(1))

kron_block = kron(chain_0, chain_1)
```

Qadence2
```python exec="on" source="material-block" html="1" session="getting_started"
from qadence2.extensions.legacy import X, Y, chain, kron

chain_0 = chain(X(0), Y(0))
chain_1 = chain(X(1), Y(1))

kron_block = kron(chain_0, chain_1)
```

## Compose Functions

Custom gates can also be applied in the same way, and with the definition change, Qadence2 now use `*` instead of `@`.

Qadence
```python exec="on" source="material-block" html="1" session="getting_started"
from qadence import X, Y, add

def xy_int(i: int, j: int):
    return (1/2) * (X(i)@X(j) + Y(i)@Y(j))

n_qubits = 3

xy_ham = add(xy_int(i, i+1) for i in range(n_qubits-1))
```

Qadence2
```python exec="on" source="material-block" html="1" session="getting_started"
from qadence2.extensions.legacy import X, Y, add

def xy_int(i: int, j: int):
    return (1/2) * (X(i)*X(j) + Y(i)*Y(j))

n_qubits = 3

xy_ham = add(xy_int(i, i+1) for i in range(n_qubits-1))
```

## Quantum Fourier Transform Example

The code initializes a quantum circuit with `CPHASE` gates and applies `qft_layer` to transform the input quantum state into its frequency domain representation.

Qadence
```python exec="on" source="material-block" html="1" session="getting_started"
from qadence import H, CPHASE, PI, chain, kron

def qft_layer(qs: tuple, l: int):
    cphases = chain(CPHASE(qs[j], qs[l], PI/2**(j-l)) for j in range(l+1, len(qs)))
    return H(qs[l]) * cphases

def qft(qs: tuple):
    return chain(qft_layer(qs, l) for l in range(len(qs)))
```

Qadence2
```python exec="on" source="material-block" html="1" session="getting_started"
from qadence2.extensions.legacy import H, CPHASE, PI, chain, kron

def qft_layer(qs: tuple, l: int):
    cphases = chain(CPHASE(qs[j], qs[l], PI/2**(j-l)) for j in range(l+1, len(qs)))
    return H(qs[l]) * cphases

def qft(qs: tuple):
    return chain(qft_layer(qs, l) for l in range(len(qs)))
```


## Quantum Models

Qadence2 uses `Expression` and `IR` to represent the details of the quantum circuits and algorithms and `Platform` to match the executing backend, whereas Qadence is not separated.
As a result, Qadence can call `run,` `sample,` and `expectation` with the circuit representation itself.
However, Qadence2 requires it first to be compiled to execute them with the backends.
Qadence2 uses `compile_to_model` and `compile_to_backend` to compile them into `model` and `compiled_model,` respectively. After this procedure, it is only ready to execute `run,` `sample,` and `expectation.`

Qadence
```python exec="on" source="material-block" html="1" session="getting_started"
import torch
from qadence import QuantumModel, PI, Z
from qadence import QuantumCircuit, RX, RY, chain, kron
from qadence import FeatureParameter, VariationalParameter

phi = FeatureParameter("phi")

block = chain(
    kron(RX(0, phi), RY(1, phi)),
)

circuit = QuantumCircuit(2, block)

observable = Z(0) + Z(1)

model = QuantumModel(circuit, observable)

values = {"phi": torch.tensor(PI)}

wf = model.run(values)
xs = model.sample(values, n_shots=100)
ex = model.expectation(values)
```

Qadence2
```python exec="on" source="material-block" html="1" session="getting_started"
import torch
from qadence2.extensions.legacy import PI, Z, RX, RY, chain, kron
from qadence2_expressions import compile_to_model, parameter
from qadence2_platforms.compiler import compile_to_backend

phi = parameter("phi")

block = chain(
    kron(RX(0, phi), RY(1, phi)),
)

model = compile_to_model(block)
compiled_model = compile_to_backend(model, "pyqtorch")

observable = Z(0) + Z(1)

values = {"phi": torch.tensor(PI)}

wf = compiled_model.run(values)
xs = compiled_model.sample(values, shots=100)
ex = compiled_model.expectation(values, observable=observable)
```

## Quantum Registers

Qadence2 can represent the relationships between qubits using `grid_type`, `grid_scale`, and `qubit_position`. The `connectivity` in `qadence2_ir` is for accurately representing the connectivity between qubits.

Qadence
```python exec="on" source="material-block" html="1" session="getting_started"
from qadence import Register

reg = Register.all_to_all(n_qubits = 4)
reg_line = Register.line(n_qubits = 4)
reg_squre = Register.square(qubits_side = 2)
reg_triang = Register.triangular_lattice(n_cells_row = 2, n_cells_col = 2)
```

Qadence2
```python exec="on" source="material-block" html="1" session="getting_started"
from qadence2_expressions.core import set_grid_type, set_qubits_positions, set_grid_scale
from qadence2.extensions.legacy import PI, RX, RY, CZ
from qadence2_expressions import compile_to_model

expr = RX(2, PI / 2) * CZ() * RY(0, PI / 2)

set_grid_type("linear")  # "square", "triangular"
set_qubits_positions([(-2, 1), (0, 0), (3, 1)])
set_grid_scale(0.4)

compile_to_model(expr)
```
