import numpy as np
import pytest

import tensorforge as tf


def test_creation_factories_and_isolation():
    source = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = tf.tensor(source)
    source[0, 0] = 99
    assert x.shape == (2, 2) and x.ndim == 2 and x.size == 4
    assert x.dtype == np.float64 and x.numpy()[0, 0] == 1
    copy = x.numpy()
    copy[0, 0] = -1
    assert x.numpy()[0, 0] == 1
    with pytest.raises(ValueError):
        x.data[0, 0] = 1
    np.testing.assert_equal(tf.zeros(2, 3).numpy(), np.zeros((2, 3)))
    np.testing.assert_equal(tf.ones((2, 3)).numpy(), np.ones((2, 3)))
    np.testing.assert_equal(tf.arange(4).numpy(), np.arange(4))
    assert tf.tensor(2).item() == 2
    tf.manual_seed(42)
    first = tf.randn(2, 3).numpy()
    tf.manual_seed(42)
    np.testing.assert_equal(first, tf.randn(2, 3).numpy())
    with pytest.raises(TypeError):
        tf.tensor([1], dtype=np.int64, requires_grad=True)
    with pytest.raises(TypeError):
        tf.tensor([1 + 2j], dtype=np.complex128)


@pytest.mark.parametrize(
    "fn",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
        lambda a, b: 3 + a * 2 - b,
        lambda a, b: 2 / a + b,
    ],
)
def test_broadcast_gradients(gradcheck, fn):
    gradcheck(fn, [np.arange(6).reshape(2, 3) + 1.0, np.array([2.0, 3.0, 4.0])])
    gradcheck(fn, [np.ones((2, 1, 3)) * 2, np.ones((1, 4, 1)) * 3])
    gradcheck(fn, [np.array(2.0), np.ones((2, 3)) * 3])


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((3,), (3,)),
        ((2, 3), (3,)),
        ((3,), (3, 2)),
        ((2, 3), (3, 4)),
        ((2, 2, 3), (3, 4)),
        ((3,), (2, 3, 4)),
        ((2, 4, 3), (3,)),
        ((2, 1, 2, 3), (1, 3, 3, 2)),
    ],
)
def test_matmul_all_ranks(gradcheck, a_shape, b_shape):
    rng = np.random.default_rng(10)
    a, b = rng.normal(size=a_shape), rng.normal(size=b_shape)
    np.testing.assert_allclose((tf.tensor(a) @ tf.tensor(b)).numpy(), a @ b)
    gradcheck(lambda a, b: a @ b, [a, b])


@pytest.mark.parametrize(
    "axis,keepdims", [(None, False), (0, False), (-1, True), ((0, 2), False), ((), False)]
)
def test_reductions(gradcheck, axis, keepdims):
    a = np.arange(12.0).reshape(2, 2, 3)
    for operation in ("sum", "mean"):

        def fn(x, operation=operation):
            return getattr(x, operation)(axis=axis, keepdims=keepdims)

        np.testing.assert_allclose(
            fn(tf.tensor(a)).numpy(), getattr(a, operation)(axis=axis, keepdims=keepdims)
        )
        gradcheck(fn, [a])


def test_shape_operations(gradcheck):
    a = np.arange(24.0).reshape(2, 3, 4)
    gradcheck(lambda x: x.reshape(6, 4).transpose(0, 1), [a])
    gradcheck(lambda x: x.permute(2, 0, 1), [a])
    gradcheck(lambda x: x.permute(-1, 0, 1), [a])
    gradcheck(lambda x: x.T, [a])
    gradcheck(lambda x: x.reshape(-1), [a])


@pytest.mark.parametrize(
    "index",
    [
        (slice(None), slice(None, None, 2)),
        ([0, 0, 1], [1, 1, 2]),
        np.array([True, False]),
        (None, slice(None), 1),
        (Ellipsis, 1),
    ],
)
def test_index_gradients(gradcheck, index):
    gradcheck(lambda x: x[index], [np.arange(6.0).reshape(2, 3)])


def test_mutable_index_is_copied():
    x = tf.tensor([1, 2, 3], requires_grad=True)
    indices = np.array([0, 0, 2])
    y = x[indices].sum()
    indices[:] = 1
    y.backward()
    np.testing.assert_equal(x.grad, [2, 0, 1])


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: x.exp(),
        lambda x: x.log(),
        lambda x: x.tanh(),
        lambda x: x.sigmoid(),
        lambda x: x.relu(),
        lambda x: x**2.5,
        lambda x: x**0,
        lambda x: x.softmax(),
        lambda x: x.log_softmax(),
    ],
)
def test_unary_gradients(gradcheck, fn):
    gradcheck(fn, [np.array([[0.2, 0.7, 1.3], [2.0, 1.0, 0.5]])])


def test_stability_and_domains():
    x = tf.tensor([-1000.0, 0.0, 1000.0], requires_grad=True)
    np.testing.assert_equal(x.sigmoid().numpy(), [0, 0.5, 1])
    y = x.softmax()
    assert np.isfinite(y.numpy()).all()
    assert y.sum().item() == pytest.approx(1)
    for fn, error in [
        (lambda: tf.tensor(0) ** -1, ZeroDivisionError),
        (lambda: tf.tensor(0) ** 0.5, ValueError),
        (lambda: tf.tensor(-1) ** 0.5, ValueError),
        (lambda: tf.tensor(0).log(), ValueError),
        (lambda: x ** tf.tensor(2), TypeError),
    ]:
        with pytest.raises(error):
            fn()


def test_shared_graph_accumulation_and_zeroing():
    x = tf.tensor(2.0, requires_grad=True)
    square = x * x
    y = square + square + x
    y.backward()
    assert x.grad == 9
    y.backward()
    assert x.grad == 18
    x.zero_grad()
    y.backward()
    assert x.grad == 9


def test_no_grad_nesting_exception_and_detach():
    x = tf.tensor(2.0, requires_grad=True)
    with tf.no_grad():
        assert not (x * x).requires_grad
        with tf.no_grad():
            assert not (x + 1).requires_grad
        assert not (x * 2).requires_grad
    with pytest.raises(RuntimeError):
        with tf.no_grad():
            raise RuntimeError("test restoration")
    assert (x * x).requires_grad
    assert not x.detach().requires_grad


def test_backward_errors_and_versions():
    with pytest.raises(RuntimeError):
        tf.tensor(1).backward()
    x = tf.tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError):
        x.backward()
    with pytest.raises(ValueError):
        x.backward([1.0])
    y = (x * x).sum()
    x._assign([2.0, 3.0])
    with pytest.raises(RuntimeError, match="changed"):
        y.backward()
    assert x.grad is None
    with pytest.raises(ValueError):
        x.sum(axis=3)
    with pytest.raises(ValueError):
        x.sum(axis=(0, 0))
    with pytest.raises(ValueError):
        tf.tensor([]).mean()


def test_deep_graph_and_float32():
    x = tf.tensor(1.0, requires_grad=True, dtype=np.float32)
    y = x
    for _ in range(1500):
        y = y + tf.tensor(1.0, dtype=np.float32)
    y.backward()
    assert y.dtype == np.float32 and x.grad == 1
