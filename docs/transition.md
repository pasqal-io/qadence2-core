
## Transition Consideration
This document is for Qadence users to use Qadence2 funcationalities with Qadence format.
The examples in Qadence Contents and Tutorials will be presented with qadence2.expressions.legacy package. The provided Operators and functions are states as below
For Qadence2 code details, please refer to https://github.com/pasqal-io/qadence2-core.

### Operators

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

### Block System

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

### Compose Functions

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

### Quantum Fourier Transform Example

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

## Block Execution

Qadence
```python exec="on" source="material-block" html="1" session="getting_started"
from qadence import kron, add, H, Z, run, sample, expectation

n_qubits = 2

# Prepares a uniform state
h_block = kron(H(i) for i in range(n_qubits))

wf = run(h_block)

xs = sample(h_block, n_shots=1000)

obs = add(Z(i) for i in range(n_qubits))
ex = expectation(h_block, obs)
```

Qadence2
```python exec="on" source="material-block" html="1" session="getting_started"
from qadence2.extensions.legacy import kron, add, H, Z
from qadence2_expressions import compile_to_model
from qadence2_platforms import OnEnum
from qadence2_platforms.compiler import compile_to_backend
import torch

n_qubits = 2

# Prepares a uniform state
h_block = kron(H(i) for i in range(n_qubits))

model = compile_to_model(h_block)
compiled_model = compile_to_backend(model, "pyqtorch")

wf = compiled_model.run()

res = compiled_model.sample(shots=1_000, on=OnEnum.EMULATOR)

obs = add(Z(i) for i in range(n_qubits))
ex = compiled_model.expectation(observable=obs)
```

## Next session

Qadence
```python exec="on" source="material-block" html="1" session="getting_started"

```

Qadence2
```python exec="on" source="material-block" html="1" session="getting_started"

```

## Next session

Qadence
```python exec="on" source="material-block" html="1" session="getting_started"

```

Qadence2
```python exec="on" source="material-block" html="1" session="getting_started"

```

## Next session

Qadence
```python exec="on" source="material-block" html="1" session="getting_started"

```

Qadence2
```python exec="on" source="material-block" html="1" session="getting_started"

```

## Next session

Qadence
```python exec="on" source="material-block" html="1" session="getting_started"

```

Qadence2
```python exec="on" source="material-block" html="1" session="getting_started"

```

## Next session

Qadence
```python exec="on" source="material-block" html="1" session="getting_started"

```

Qadence2
```python exec="on" source="material-block" html="1" session="getting_started"

```
