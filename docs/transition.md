
### Transition Consideration
This document is for Qadence users to use Qadence2 funcationalities with Qadence format.
The examples in Qadence Contents and Tutorials will be presented with qadence2.legacy.
For Qadence2 code manual, please refer to ~~.

Qadence Contents
- Block System
- Parametric Programs
- Quantum Models
- Quantum Registers
- State Initializaton
- Arbitary Hamiltonians
- Time-dependent Generators
- QML Constructors
- Wavefunction Overlaps
- Backends

Qadence Tutorials
- Digital-Analog Quantum Computation
- Basic operations on neutral-atoms ...


# Block System

## Primitive blocks

Qadence
```python exec="on" source="material-block" html="1" session="getting_started"
from qadence import chain, RX, CNOT

rx = RX(0, 0.5)
cnot = CNOT(0, 1)

block = chain(rx, cnot)
```

Qadence2
```python exec="on" source="material-block" html="1" session="getting_started"
from qadence2.extensions.legacy import chain, RX, CNOT

rx = RX(0, 0.5)
cnot = CNOT(0, 1)

block = chain(rx, cnot)
```

## Composite blocks

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
    return (1/2) * (X(i)@X(j) + Y(i)@Y(j))

n_qubits = 3

xy_ham = add(xy_int(i, i+1) for i in range(n_qubits-1))
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
