This document is for Qadence users to easily adopte to Qadence 2.
It is separated into several parts;

## Primitive blocks

A [`PrimitiveBlock`][qadence.blocks.primitive.PrimitiveBlock] represents a digital or an analog time-evolution quantum operation applied to a qubit support. Programs can always be decomposed down into a sequence of `PrimitiveBlock` elements.

Two canonical examples of digital primitive blocks are the parametrized `RX` and the `CNOT` gates:

```python exec="on" source="material-block" html="1" session="getting_started"
from qadence import chain, RX, CNOT

rx = RX(0, 0.5)
cnot = CNOT(0, 1)

block = chain(rx, cnot)

from qadence.draw import html_string # markdown-exec: hide
print(html_string(block)) # markdown-exec: hide
```

## Transition Guide Consideration

What would Qadence users face or need when they try to use Qadence2?
In general, to maximize Qadence2's objective, users can program the quantum algorithms using only the highest level of language. -> Qadence2-Expressions.
For some users to want to manipulate the tool in depth, functionalities from Qadence2-IR and Qadence2-Platforms are also provided.

## Using Qadence2-Expressions

Qadence2-Expressions use `parameter`, `variable`.


What to show example in Qadence
- Quantum Register
- Quantum Models
- Execution
- State Initialization
- Parametric Programs

### Quantum Register Transition

Qadence
```python exec="on"
from qadence import Register

reg = Register.all_to_all(n_qubits = 4)
reg_line = Register.line(n_qubits = 4)
reg_circle = Register.circle(n_qubits = 4)
reg_squre = Register.square(qubits_side = 2)
reg_rect = Register.rectangular_lattice(qubits_row = 2, qubits_col = 2)
reg_triang = Register.triangular_lattice(n_cells_row = 2, n_cells_col = 2)
reg_honey = Register.honeycomb_lattice(n_cells_row = 2, n_cells_col = 2)
```

Qadence2
```python exec="on"
# No explicit declaration for number of qubits is required in expression level. Qubit mapping will be handled automatically.
# But you can set the formation of qubits in IR level.
```

### Transition Guide Consideration
Since Qadence is a single tool and Qadence2 is separated into four parts, but working in a whole, it seems needless to match which function in Qadence is transferred to some function in Qadence2. Thus, I'll first map every example in Qadence Contents in identical functionality in Qadence2.

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


### Block System

Qadence
```python exec="on"
from qadence import chain, RX, CNOT

rx = RX(0, 0.5)
cnot = CNOT(0, 1)

block = chain(rx, cnot)
```

Qadence2
```python exec="on"
from qadence2-expressions import RX, CNOT

rx = RX()
```