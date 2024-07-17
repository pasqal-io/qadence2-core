from __future__ import annotations

import torch
from qadence2_ir import Model

from qadence2.expressions import RX, RY, Z, irc, parameter, variable
from qadence2.expressions.expression import visualize_expression
from qadence2.platforms import compile as compile_ir
from qadence2.platforms.backend.api import Api

# Abstraction Levels

## Level 0: Expression

### Define our root parameters, trainable and non-trainable
x = parameter("x")
y = parameter("y")
theta = variable("theta")

### For parametric gates, we construct a closure over our parameter expression and call that on a support
expr = RX(x * theta / 2)(2) * RY(y * theta / 2)(0)
### An optional observable
obs_expr = Z(0)
print(visualize_expression(expr))

## Level 1: Intermediate Representation
ir = irc(
    expr
)  ### This returns a `Model` which a collection of data relating to our abstract expression
print(ir)


## Level 2: API
def compile(expr, backend_name, obs_expr=None) -> Api:
    ir: Model = irc(expr)
    obs_ir = irc(obs_expr) if obs_expr else None
    return compile_ir(ir, backend_name, obs_ir)


### This returns the equivalent of `QuantumModel` which has run, sample and expectation methods
api = compile(expr, "pyqtorch", obs_expr)
### The native sequence, which can be a sequence of digital gates/ pulses or a mix of digital and analog operations
print(api.sequence)
### The embedding holds the randomly initialized variational parameters
vparams = api.embedding.vparams
### If we compile to pyqtorch, we need to the values for our feature parameter 'x' and 'y' as torch.Tensors
inputs = {"x": torch.rand(1), "y": torch.rand(1)}

### Let's store them in a single dict
root_parameters = {**vparams, **inputs}

### Now we can call the api object using the legacy qadence syntax
wf = api.run(inputs=root_parameters)
samples = api.sample(inputs=root_parameters, n_shots=1000)[0]
expval = api.expectation(inputs=root_parameters)
print(wf)
print(samples)
print(expval)
