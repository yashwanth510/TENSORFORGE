"""Run: python examples/01_scalar_autograd.py"""

from tensorforge import Value

# A scalar is one number. Reusing a creates two paths to the output.
a, b = Value(2), Value(3)
loss = a * b + a
loss.backward()
print(f"loss={loss.data}, dloss/da={a.grad}, dloss/db={b.grad}")
assert (loss.data, a.grad, b.grad) == (8.0, 4.0, 2.0)
