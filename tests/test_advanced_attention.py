import numpy as np
import pytest

import tensorforge as tf
from tensorforge.attention import RelativePositionBias, alibi_bias, memory_efficient_attention
from tensorforge.nn import RotaryEmbedding
from tensorforge.nn.functional import scaled_dot_product_attention as dense


@pytest.mark.parametrize("causal", [False, True])
def test_tiled_attention_values_and_gradients(gradcheck, causal):
    rng = np.random.default_rng(12)
    q, k, v = [rng.normal(size=(1, 2, 3, 2)) for _ in range(3)]
    mask = np.zeros((3, 3), dtype=bool)
    mask[2, :] = True

    def function(q, k, v):
        return memory_efficient_attention(q, k, v, block_size=2, is_causal=causal, mask=mask)

    inputs = [tf.tensor(a) for a in (q, k, v)]
    np.testing.assert_allclose(
        function(*inputs).numpy(), dense(*inputs, mask=mask, is_causal=causal).numpy(), atol=1e-12
    )
    gradcheck(function, [q, k, v])


def test_local_attention_blocks_old_and_future_tokens():
    q, k, v = (
        tf.ones(1, 1, 4, 2),
        tf.ones(1, 1, 4, 2),
        tf.tensor(np.arange(8.0).reshape(1, 1, 4, 2)),
    )
    result = memory_efficient_attention(q, k, v, window_size=2, is_causal=True).numpy()
    np.testing.assert_allclose(result[0, 0], [[0, 1], [1, 2], [3, 4], [5, 6]])


def test_bias_gradients_and_rope_scaling(gradcheck):
    bias = alibi_bias(3, 4)
    assert bias.shape == (3, 4, 4)
    np.testing.assert_equal(np.diagonal(bias.numpy(), axis1=-2, axis2=-1), 0)
    relative = RelativePositionBias(2, 2)
    x = tf.ones(1, 2, 3, 2)
    dense(x, x, x, bias=relative(3)).sum().backward()
    assert relative.weight.grad is not None
    q = np.arange(12.0).reshape(1, 2, 3, 2) / 10
    gradcheck(
        lambda b: dense(tf.tensor(q), tf.tensor(q), tf.tensor(q), bias=b), [np.zeros((2, 3, 3))]
    )
    x = tf.ones(1, 1, 1, 4)
    np.testing.assert_allclose(
        RotaryEmbedding(4, scale=2)(x, offset=6).numpy(), RotaryEmbedding(4)(x, offset=3).numpy()
    )


def test_paged_cache_eviction_reuse_and_atomic_capacity():
    from tensorforge.cache import PagedKVCache

    cache = PagedKVCache(3, 2, 1, 2)
    keys = np.arange(10.0).reshape(1, 5, 2)
    cache.append("a", keys, keys + 1)
    before = cache.get("a")[0].numpy()
    with pytest.raises(MemoryError):
        cache.append("b", np.ones((1, 1, 2)), np.ones((1, 1, 2)))
    assert "b" not in cache.requests
    np.testing.assert_equal(cache.get("a")[0].numpy(), before)
    cache.evict("a", keep_last=2)
    k, v, positions = cache.get("a")
    np.testing.assert_equal(k.numpy(), keys[:, 3:])
    np.testing.assert_equal(positions, [3, 4])
    cache.append("b", np.ones((1, 2, 2)), np.ones((1, 2, 2)))
    cache.release("a")
    cache.release("b")
    assert len(cache.free) == 3


def test_cache_empty_eviction_and_overflow_are_safe():
    from tensorforge.cache import PagedKVCache

    cache = PagedKVCache(1, 4, 1, 2, dtype=np.float32)
    item = np.ones((1, 1, 2))
    cache.append("a", item, item)
    cache.evict("a", keep_last=0)
    assert len(cache.free) == 1
    assert cache.get("a")[0].shape == (1, 0, 2)
    with pytest.raises(ValueError, match="overflow"):
        cache.append("b", item * 1e100, item)
    assert len(cache.free) == 1 and "b" not in cache.requests
    cache.append("a", item * 2, item)
    np.testing.assert_equal(cache.get("a")[2], [1])
    np.testing.assert_equal(cache.get("a")[0].numpy(), item * 2)


def test_tiled_attention_requires_float_inputs():
    integers = tf.tensor([[1, 2]], dtype=np.int64)
    with pytest.raises(TypeError, match="floating point"):
        memory_efficient_attention(integers, integers, integers)
