import numpy as np
import pytest

import tensorforge as tf


@pytest.mark.parametrize(
    "function",
    [
        lambda x: x.clone(),
        lambda x: x.abs(),
        lambda x: x.sin(),
        lambda x: x.cos(),
        lambda x: x.gelu(),
        lambda x: x.silu(),
        lambda x: x.unsqueeze(0).squeeze(0),
        lambda x: x.flatten(),
        lambda x: x.amax(axis=1),
        lambda x: x.amin(axis=0),
        lambda x: x.amax(axis=(0, 1), keepdims=True),
    ],
)
def test_new_operations(gradcheck, function):
    gradcheck(function, [np.array([[-1.3, 0.5, 2.4], [3.1, -0.7, 1.2]])])


def test_expand_cat_stack_and_where(gradcheck):
    gradcheck(lambda a, b: tf.cat((a, b), axis=-1), [np.ones((2, 2)), np.ones((2, 3))])
    gradcheck(lambda a, b: tf.stack((a, b), axis=1), [np.ones((2, 3)), np.ones((2, 3)) * 2])
    gradcheck(lambda a: a.expand(2, 4, 3), [np.ones((1, 3))])
    mask = np.array([[True, False, True], [False, True, False]])
    gradcheck(lambda a, b: tf.where(mask, a, b), [np.array([1.0, 2.0, 3.0]), np.array(2.0)])
    gradcheck(lambda a: a.masked_fill(mask, 2), [np.ones((2, 3))])
    with pytest.raises(TypeError):
        tf.where([1, 0], tf.ones(2), tf.zeros(2))
    with pytest.raises(ValueError):
        tf.stack([tf.ones(2), tf.ones(3)])


def test_max_ties_and_argmax():
    x = tf.tensor([[2, 2, 1], [0, 3, 3]], requires_grad=True)
    x.amax(axis=1).sum().backward()
    np.testing.assert_equal(x.grad, [[0.5, 0.5, 0], [0, 0.5, 0.5]])
    result = x.argmax(axis=1)
    np.testing.assert_equal(result.numpy(), [0, 1])
    assert result.dtype == np.int64 and not result.requires_grad


def test_rng_state_roundtrip():
    tf.manual_seed(34)
    state = tf.get_rng_state()
    a = tf.randn(3).numpy()
    tf.set_rng_state(state)
    np.testing.assert_equal(a, tf.randn(3).numpy())
