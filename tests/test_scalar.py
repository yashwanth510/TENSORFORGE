"""Learn the smallest graph first; pytest finds functions named test_* automatically."""

import math

import pytest

from tensorforge import Value


def test_creation_and_repr():
    x = Value(2)
    assert x.data == 2.0 and x.grad == 0.0
    assert repr(x) == "Value(data=2.0, grad=0.0)"


def test_shared_paths():
    a, b = Value(2), Value(3)
    loss = a * b + a
    loss.backward()
    assert loss.data == 8
    assert a.grad == 4 and b.grad == 2


def test_shared_intermediate_and_repeat():
    a = Value(2)
    square = a * a
    loss = square + square
    loss.backward()
    assert a.grad == 8
    loss.backward()
    assert a.grad == 8  # Scalar API deliberately resets each pass.


@pytest.mark.parametrize(
    "fn, expected, derivative",
    [
        (lambda x: 3 + x, 5, 1),
        (lambda x: 3 * x, 6, 3),
        (lambda x: 3 - x, 1, -1),
        (lambda x: x - 3, -1, 1),
        (lambda x: 8 / x, 4, -2),
        (lambda x: x / 4, 0.5, 0.25),
        (lambda x: x**3, 8, 12),
        (lambda x: -x, -2, -1),
        (lambda x: x.exp(), math.exp(2), math.exp(2)),
        (lambda x: x.log(), math.log(2), 0.5),
        (lambda x: x.tanh(), math.tanh(2), 1 - math.tanh(2) ** 2),
    ],
)
def test_operations(fn, expected, derivative):
    x = Value(2)
    result = fn(x)
    result.backward()
    assert result.data == pytest.approx(expected)
    assert x.grad == pytest.approx(derivative)


@pytest.mark.parametrize("number, output, grad", [(-2, 0, 0), (0, 0, 0), (2, 2, 1)])
def test_relu(number, output, grad):
    x = Value(number)
    result = x.relu()
    result.backward()
    assert result.data == output and x.grad == grad


def test_zero_power_and_domains():
    x = Value(0)
    (x**0).backward()
    assert x.grad == 0
    for fn, error in [
        (lambda: x**-1, ZeroDivisionError),
        (lambda: x**0.5, ValueError),
        (lambda: Value(-2) ** 0.5, ValueError),
        (lambda: x ** Value(2), TypeError),
        (lambda: x.log(), ValueError),
    ]:
        with pytest.raises(error):
            fn()


def test_deep_graph():
    x = Value(1)
    result = x
    for _ in range(1500):
        result = result + 1
    result.backward()
    assert result.data == 1501 and x.grad == 1


def test_forward_values_are_saved():
    x = Value(3)
    y = x * x
    x.data = 100
    y.backward()
    assert x.grad == 6
