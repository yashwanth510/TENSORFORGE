import numpy as np
import pytest

import tensorforge as tf
from tensorforge import nn, optim
from tensorforge.finetuning import (
    Adapter,
    LoRALinear,
    QuantizedLinear,
    dpo_loss,
    policy_gradient_loss,
    reward_ranking_loss,
    sequence_log_probs,
    trainable_parameters,
)


def test_lora_updates_only_adapter_and_merges():
    tf.manual_seed(4)
    layer = LoRALinear(nn.Linear(2, 1), rank=2, alpha=2)
    original = layer.base.weight.numpy()
    x = tf.tensor([[1.0, 2.0], [2.0, 3.0], [-1.0, 1.0]])
    target = tf.tensor([[1.0], [2.0], [0.0]])
    np.testing.assert_equal(layer(x).numpy(), layer.base(x).numpy())
    optimizer = optim.Adam(trainable_parameters(layer), lr=0.03)
    initial = nn.MSELoss()(layer(x), target).item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = nn.MSELoss()(layer(x), target)
        loss.backward()
        optimizer.step()
    assert nn.MSELoss()(layer(x), target).item() < initial * 0.2
    np.testing.assert_equal(layer.base.weight.numpy(), original)
    assert layer.base.weight.grad is None
    np.testing.assert_allclose(layer.merged()(x).numpy(), layer(x).numpy(), atol=1e-12)


def test_adapter_and_quantization(gradcheck):
    x = np.array([[1.0, 2.0], [-2.0, 3.0]])
    adapter = Adapter(2, 1)
    np.testing.assert_equal(adapter(tf.tensor(x)).numpy(), x)
    gradcheck(adapter, [x])
    layer = nn.Linear(2, 3)
    quantized = QuantizedLinear(layer)
    assert quantized.parameters() == []
    assert quantized.weight_int8.dtype == np.int8
    error = np.max(np.abs(quantized(tf.tensor(x)).numpy() - layer(tf.tensor(x)).numpy()))
    assert error <= np.max(quantized.scale.numpy()) * np.max(np.abs(x).sum(axis=1))
    gradcheck(quantized, [x])


def test_preference_and_policy_gradients(gradcheck):
    a, b = np.array([-1.0, -2.0]), np.array([-3.0, -4.0])
    gradcheck(lambda a, b: dpo_loss(a, b, np.array([-2.0, -2.0]), np.array([-3.0, -3.0])), [a, b])
    gradcheck(reward_ranking_loss, [a, b])
    ref = tf.tensor(a, requires_grad=True)
    chosen = tf.tensor(a, requires_grad=True)
    dpo_loss(chosen, tf.tensor(b), ref, tf.tensor(b)).backward()
    assert ref.grad is None and np.all(chosen.grad < 0)
    logp = tf.tensor([-1.0, -2.0], requires_grad=True)
    advantages = tf.tensor([1.0, -1.0], requires_grad=True)
    policy_gradient_loss(logp, advantages).backward()
    assert advantages.grad is None
    np.testing.assert_equal(logp.grad, [-0.5, 0.5])


def test_sequence_log_probs(gradcheck):
    x = np.arange(12.0).reshape(2, 2, 3) / 10
    targets = np.array([[0, 1], [2, -100]])
    gradcheck(lambda x: sequence_log_probs(x, targets), [x])
    values = sequence_log_probs(tf.tensor(x), targets).numpy()
    lp = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
    np.testing.assert_allclose(values, [lp[0, 0, 0] + lp[0, 1, 1], lp[1, 0, 2]])


def test_checkpoint_recomputation_and_rng():
    from tensorforge.training import checkpoint

    tf.manual_seed(31)
    model = nn.Sequential(nn.Linear(2, 3), nn.Dropout(0.4), nn.Tanh(), nn.Linear(3, 1))
    x = tf.tensor([[1.0, 2.0], [-1.0, 3.0]], requires_grad=True)
    state = tf.get_rng_state()
    reference = model(x)
    reference.sum().backward()
    expected = [p.grad.copy() for p in model.parameters()]
    dx = x.grad.copy()
    model.zero_grad()
    x.zero_grad()
    tf.set_rng_state(state)
    replay = checkpoint(model, x)
    np.testing.assert_equal(replay.numpy(), reference.numpy())
    after = tf.get_rng_state()
    replay.sum().backward()
    assert tf.get_rng_state() == after
    for p, g in zip(model.parameters(), expected):
        np.testing.assert_equal(p.grad, g)
    np.testing.assert_equal(x.grad, dx)
    with pytest.raises(ValueError):
        checkpoint(nn.BatchNorm1d(2), x)
