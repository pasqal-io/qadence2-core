

```python
from qadence2.expressions import RX, RY, parameter, variable, irc
from qadence2.platforms import compile

a = parameter("a")
b = parameter("b")
phi = variable("phi")
expr = RX(a * phi / 2)(2) * RY(b * phi / 2)(0)

irc(expr)
```
