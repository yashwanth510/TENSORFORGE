"""Run: python examples/02_tensor_basics.py"""

import numpy as np

import tensorforge as tf

# Two rows and three columns: shape (2, 3).
x = tf.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
bias = tf.tensor([10, 20, 30])
print("Shape:", x.shape)
print("Broadcast addition:\n", (x + bias).numpy())
print("Column sums:", x.sum(axis=0).numpy())
print("Reshape:\n", x.reshape(3, 2).numpy())
print("Transpose:\n", x.T.numpy())
print("Matrix multiplication:\n", (x @ x.T).numpy())

# The sum of squares has derivative 2*x at every entry.
loss = (x * x).sum()
loss.backward()
print("Gradient:\n", x.grad)
np.testing.assert_equal(x.grad, 2 * x.numpy())
