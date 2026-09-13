import numpy as np
import pytest

import tensorforge as tf
from tensorforge import nn
from tensorforge.models import CausalLanguageModel, LanguageModelConfig
from tensorforge.nn.functional import scaled_dot_product_attention as attention


def test_attention_mask_values_and_gradients(gradcheck):
    rng = np.random.default_rng(1)
    q, k, v = [rng.normal(size=(1, 1, 3, 2)) for _ in range(3)]
    mask = np.array([[False, True, True], [False, False, True], [True, True, True]])
    gradcheck(lambda q, k, v: attention(q, k, v, mask=mask), [q, k, v])
    result = attention(tf.tensor(q), tf.tensor(k), tf.tensor(v), mask=mask).numpy()
    np.testing.assert_allclose(result[0, 0, 0], v[0, 0, 0])
    np.testing.assert_equal(result[0, 0, 2], 0)
    with pytest.raises(TypeError):
        attention(tf.tensor(q), tf.tensor(k), tf.tensor(v), mask=np.ones((3, 3)))


def test_attention_causal_no_future_leakage():
    tf.manual_seed(8)
    layer = nn.MultiheadAttention(8, 2)
    x = tf.randn(1, 4, 8)
    changed = x.numpy()
    changed[:, 2:] += 100
    a = layer(x, is_causal=True).numpy()
    b = layer(tf.tensor(changed), is_causal=True).numpy()
    np.testing.assert_allclose(a[:, :2], b[:, :2])
    padded = np.array([[False, False, True, True]])
    a = layer(x, key_padding_mask=padded).numpy()
    b = layer(x, tf.tensor(changed), tf.tensor(changed), key_padding_mask=padded).numpy()
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("rope,kv_heads", [(False, None), (True, 2), (True, 1)])
def test_multihead_gradient_and_cache(gradcheck, rope, kv_heads):
    tf.manual_seed(2)
    layer = nn.MultiheadAttention(4, 2, rope=rope, num_kv_heads=kv_heads)
    x = np.arange(12.0).reshape(1, 3, 4) / 13
    gradcheck(lambda x: layer(x, is_causal=True), [x])
    layer.eval()
    with tf.no_grad():
        full = layer(tf.tensor(x), is_causal=True)
        first, cache = layer(tf.tensor(x[:, :2]), is_causal=True, use_cache=True)
        last, cache = layer(tf.tensor(x[:, 2:]), is_causal=True, cache=cache, use_cache=True)
        np.testing.assert_allclose(tf.cat((first, last), axis=1).numpy(), full.numpy(), atol=1e-12)
        assert cache[0].shape == (1, 2 if kv_heads is None else kv_heads, 3, 2)
    with pytest.raises(RuntimeError):
        layer(tf.tensor(x), use_cache=True)


def test_rotary_preserves_norm_and_position_tables(gradcheck):
    rope = nn.RotaryEmbedding(4)
    x = np.arange(24.0).reshape(1, 2, 3, 4) / 7
    gradcheck(lambda x: rope(x, offset=3), [x])
    np.testing.assert_allclose((rope(tf.tensor(x), 3).numpy() ** 2).sum(-1), (x * x).sum(-1))
    position = nn.SinusoidalPositionEncoding(5, 10)
    result = position(tf.zeros(1, 3, 5))
    np.testing.assert_equal(result.numpy()[0, 0], [0, 1, 0, 1, 0])
    learned = nn.LearnedPositionEncoding(4, 10)
    learned(tf.zeros(1, 3, 4), offset=2).sum().backward()
    np.testing.assert_equal(learned.embedding.weight.grad[2:5], 1)


def test_encoder_decoder_gradients_and_padding():
    tf.manual_seed(4)
    network = nn.Transformer(4, 2, hidden_size=8)
    source, target = tf.randn(2, 3, 4, requires_grad=True), tf.randn(2, 2, 4, requires_grad=True)
    loss = (network(source, target) ** 2).mean()
    loss.backward()
    assert np.isfinite(source.grad).all() and np.isfinite(target.grad).all()
    assert all(p.grad is not None and np.isfinite(p.grad).all() for p in network.parameters())


@pytest.mark.parametrize("variant,kv_heads", [("gpt", None), ("modern", 1)])
def test_language_model_cache_and_causality(variant, kv_heads):
    tf.manual_seed(7)
    config = LanguageModelConfig(
        9,
        features=8,
        num_heads=2,
        num_layers=2,
        hidden_size=12,
        max_length=12,
        variant=variant,
        num_kv_heads=kv_heads,
    )
    model = CausalLanguageModel(config)
    ids = tf.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int64)
    model.eval()
    with tf.no_grad():
        full = model(ids)
        prefix, cache = model(ids[:, :2], use_cache=True)
        tail, cache = model(ids[:, 2:], cache=cache, use_cache=True)
        np.testing.assert_allclose(tf.cat((prefix, tail), axis=1).numpy(), full.numpy(), atol=1e-12)
        changed = ids.numpy()
        changed[:, 2:] = 8
        np.testing.assert_allclose(model(changed).numpy()[:, :2], full.numpy()[:, :2])
    greedy = model.generate(ids, max_new_tokens=3, temperature=0, use_cache=True)
    uncached = model.generate(ids, max_new_tokens=3, temperature=0, use_cache=False)
    np.testing.assert_equal(greedy.numpy(), uncached.numpy())
    a = model.generate(ids, max_new_tokens=3, top_k=3, top_p=0.8, seed=7)
    b = model.generate(ids, max_new_tokens=3, top_k=3, top_p=0.8, seed=7, use_cache=False)
    np.testing.assert_equal(a.numpy(), b.numpy())
    assert not model.training
    model.train()
    model.layers[0].eval()
    model.generate(ids, max_new_tokens=0)
    assert model.training and not model.layers[0].training


def test_generation_eos_and_validation():
    config = LanguageModelConfig(
        4, features=4, num_heads=1, num_layers=1, hidden_size=8, max_length=8
    )
    model = CausalLanguageModel(config)
    for p in model.parameters():
        p._assign(np.zeros(p.shape))
    prompt = np.array([[1], [2]])
    result = model.generate(prompt, max_new_tokens=5, temperature=0, eos_token_id=0)
    assert result.shape == (2, 2)
    assert model.training
    for kwargs in ({"top_k": 0}, {"top_p": 0}, {"temperature": -1}, {"max_new_tokens": 9}):
        with pytest.raises(ValueError):
            model.generate(prompt, **kwargs)
    with pytest.raises(TypeError):
        model([[1.5]])
    with pytest.raises(ValueError):
        LanguageModelConfig(4, features=6, num_heads=2, variant="modern")
