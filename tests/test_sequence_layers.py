import numpy as np
import pytest

import tensorforge as tf
from tensorforge import nn
from tensorforge.nn import functional as F


@pytest.mark.parametrize("layer", [nn.RNN, nn.LSTM, nn.GRU])
def test_recurrent_input_and_state_gradients(gradcheck, layer):
    tf.manual_seed(3)
    network = layer(2, 2)
    rng = np.random.default_rng(5)
    x, h, c = rng.normal(size=(1, 3, 2)), rng.normal(size=(1, 2)), rng.normal(size=(1, 2))
    if layer is nn.LSTM:
        gradcheck(lambda x, h, c: network(x, (h, c))[0], [x, h, c])
    else:
        gradcheck(lambda x, h: network(x, h)[0], [x, h])
    network.zero_grad()
    output, _ = network(tf.tensor(x))
    output.sum().backward()
    assert all(p.grad is not None and np.isfinite(p.grad).all() for p in network.parameters())


@pytest.mark.parametrize("layer", [nn.RNN, nn.LSTM, nn.GRU])
def test_recurrent_chunk_continuation(layer):
    tf.manual_seed(9)
    network = layer(2, 3)
    x = tf.randn(2, 5, 2)
    full, _ = network(x)
    first, state = network(x[:, :2])
    second, _ = network(x[:, 2:], state)
    np.testing.assert_allclose(tf.cat((first, second), axis=1).numpy(), full.numpy())
    with pytest.raises(ValueError):
        network(tf.ones(2, 0, 2))


def test_norms_and_buffers(gradcheck):
    x = np.array([[0.4, 0.7, -0.8], [1.2, -0.9, 0.1], [0.3, 0.6, 2.1]])
    gradcheck(nn.RMSNorm(3), [x])
    layer = nn.BatchNorm1d(3, momentum=0.5)
    layer(tf.tensor(x))
    np.testing.assert_allclose(layer.running_mean.numpy(), 0.5 * x.mean(axis=0))
    np.testing.assert_allclose(layer.running_var.numpy(), 0.5 + 0.5 * x.var(axis=0, ddof=1))
    assert layer.num_batches_tracked.item() == 1
    assert len(layer.buffers()) == 3 and len(layer.parameters()) == 2
    clone = nn.BatchNorm1d(3)
    clone.load_state_dict(layer.state_dict())
    clone.eval()
    expected = (x - layer.running_mean.numpy()) / np.sqrt(layer.running_var.numpy() + layer.eps)
    np.testing.assert_allclose(clone(tf.tensor(x)).numpy(), expected)
    gradcheck(clone, [x])
    training = nn.BatchNorm1d(3, momentum=0)
    gradcheck(training, [x])
    image = nn.BatchNorm2d(2)
    assert image(tf.randn(2, 2, 3, 3)).shape == (2, 2, 3, 3)


def test_norm_validation_and_collections():
    with pytest.raises(ValueError):
        nn.BatchNorm1d(2)(tf.ones(1, 2))
    model = nn.ModuleDict({"a": nn.ModuleList([nn.Linear(2, 3), nn.Linear(3, 1)])})
    assert len(model.parameters()) == 4
    model.eval()
    assert not model["a"][0].training
    with pytest.raises(ValueError):
        model.register_buffer("training", tf.zeros(1))
    model.register_buffer("counter", tf.tensor(0, dtype=np.int64))
    assert "counter" in model.state_dict()


def test_average_pooling_and_integer_conv(gradcheck):
    x = np.arange(16.0).reshape(1, 1, 4, 4)
    gradcheck(nn.AvgPool2d(2, stride=1), [x])
    result = nn.AvgPool2d(2)(tf.tensor(x))
    np.testing.assert_allclose(result.numpy(), [[[[2.5, 4.5], [10.5, 12.5]]]])
    layer = nn.Conv2d(1, 2, 2)
    layer(tf.tensor(x, dtype=np.int64)).sum().backward()
    assert np.isfinite(layer.weight.grad).all()


def test_sequence_loss_ignore_and_smoothing(gradcheck):
    x = np.arange(12.0).reshape(2, 2, 3) / 5
    targets = np.array([[0, 1], [-100, 2]])
    gradcheck(lambda x: F.cross_entropy(x, targets, label_smoothing=0.2), [x])
    logits = tf.tensor(x, requires_grad=True)
    loss = F.cross_entropy(logits, targets, reduction="none", label_smoothing=0.2)
    assert loss.shape == (2, 2) and loss.numpy()[1, 0] == 0
    loss.sum().backward()
    np.testing.assert_equal(logits.grad[1, 0], 0)
    ignored = tf.tensor(x, requires_grad=True)
    result = F.cross_entropy(ignored, np.full((2, 2), -100))
    result.backward()
    assert result.item() == 0
    np.testing.assert_equal(ignored.grad, np.zeros_like(x))
    with pytest.raises(ValueError):
        F.cross_entropy(logits, targets, label_smoothing=2)
    gradcheck(lambda x: nn.L1Loss()(x, tf.zeros(2, 2, 3)), [x + 0.1])
