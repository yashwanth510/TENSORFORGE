"""Independent central finite differences are the oracle for our derivatives."""

import numpy as np
import pytest

from tensorforge import Tensor


@pytest.fixture
def gradcheck():
    def check(function, arrays, atol=2e-5, rtol=2e-4):
        arrays = [np.array(a, dtype=np.float64, copy=True) for a in arrays]
        inputs = [Tensor(a, requires_grad=True) for a in arrays]
        output = function(*inputs)
        # Nonuniform upstream values test more than the special all-ones case.
        upstream = np.linspace(0.3, 1.3, output.size).reshape(output.shape)
        output.backward(upstream)
        for index, (array, tensor) in enumerate(zip(arrays, inputs)):
            expected = np.empty_like(array)
            for position in np.ndindex(array.shape):
                plus, minus = [a.copy() for a in arrays], [a.copy() for a in arrays]
                plus[index][position] += 1e-6
                minus[index][position] -= 1e-6
                yp = function(*[Tensor(a) for a in plus]).numpy()
                ym = function(*[Tensor(a) for a in minus]).numpy()
                expected[position] = np.sum((yp - ym) * upstream) / (2e-6)
            np.testing.assert_allclose(tensor.grad, expected, atol=atol, rtol=rtol)

    return check
