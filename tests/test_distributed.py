import numpy as np

import tensorforge as tf
from tensorforge import nn
from tensorforge.distributed import data_parallel_backward
from tensorforge.nn.functional import mse_loss


def test_cpu_parallel_gradient_average():
    tf.manual_seed(3)
    model = nn.Linear(2, 1)
    x, y = tf.randn(5, 2), tf.randn(5, 1)
    loss = mse_loss(model(x), y)
    loss.backward()
    expected = [p.grad.copy() for p in model.parameters()]
    model.zero_grad()
    parallel = data_parallel_backward(model, [(x[:2], y[:2]), (x[2:], y[2:])], mse_loss, workers=2)
    np.testing.assert_allclose(parallel, loss.item())
    for p, g in zip(model.parameters(), expected):
        np.testing.assert_allclose(p.grad, g, atol=1e-12)
