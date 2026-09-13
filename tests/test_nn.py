import numpy as np
import pytest

import tensorforge as tf
from tensorforge import nn
from tensorforge.nn import functional as F


def test_linear_and_registration():
    tf.manual_seed(1)
    layer = nn.Linear(3, 2)
    x = tf.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = layer(x)
    np.testing.assert_allclose(y.numpy(), x.numpy() @ layer.weight.numpy().T + layer.bias.numpy())
    y.sum().backward()
    np.testing.assert_allclose(x.grad, layer.weight.numpy().sum(axis=0, keepdims=True))
    np.testing.assert_equal(layer.weight.grad, [[1, 2, 3], [1, 2, 3]])
    np.testing.assert_equal(layer.bias.grad, [1, 1])
    model = nn.Sequential(layer, nn.ReLU(), layer)
    assert len(model.parameters()) == 2  # Shared layers are registered once.
    assert list(model.state_dict()) == ["layers.0.weight", "layers.0.bias"]
    model.eval()
    assert not layer.training
    model.train()
    assert layer.training
    model.zero_grad()
    assert all(p.grad is None for p in model.parameters())


def test_custom_module_containers_and_cycles():
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = {"first": nn.Linear(2, 2)}
            self.shared = self.layers["first"]
            self.cycle = self

        def forward(self, x):
            return self.layers["first"](x)

    model = Network()
    assert len(model.parameters()) == 2
    assert model(tf.ones(1, 2)).shape == (1, 2)
    model.eval()
    assert not model.shared.training


def test_losses_and_extreme_logits(gradcheck):
    prediction = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[0.0, 3.0], [2.0, 5.0]])
    gradcheck(lambda x, y: F.mse_loss(x, y), [prediction, target])
    labels = np.array([0, 2], dtype=np.int64)
    logits = np.array([[1.0, 2.0, 3.0], [0.2, -0.3, 0.8]])
    gradcheck(lambda x: F.cross_entropy(x, labels), [logits])
    gradcheck(
        lambda x, y: F.binary_cross_entropy_with_logits(x, y),
        [np.array([-2.0, 0.0, 3.0]), np.array([0.2, 0.5, 0.8])],
    )
    x = tf.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]], requires_grad=True)
    loss = F.cross_entropy(x, np.array([1, 0]))
    assert loss.item() == pytest.approx(2000)
    loss.backward()
    assert np.isfinite(x.grad).all()
    b = tf.tensor([-1000.0, 0.0, 1000.0], requires_grad=True)
    loss = nn.BCEWithLogitsLoss()(b, tf.tensor([1.0, 0.0, 0.0]))
    loss.backward()
    np.testing.assert_allclose(b.grad, [-1 / 3, 1 / 6, 1 / 3])
    assert F.mse_loss(tf.tensor(prediction), tf.tensor(target), "none").shape == (2, 2)
    assert F.mse_loss(tf.tensor(prediction), tf.tensor(target), "sum").item() == 4


@pytest.mark.parametrize(
    "call",
    [
        lambda: F.mse_loss(tf.ones(2, 1), tf.ones(2)),
        lambda: F.mse_loss(tf.ones(2), tf.ones(2), "invalid"),
        lambda: F.cross_entropy(tf.ones(2, 3), np.array([0.0, 1.0])),
        lambda: F.cross_entropy(tf.ones(2, 3), np.array([0, 3])),
        lambda: F.cross_entropy(tf.ones(2, 3), np.array([-1, 1])),
        lambda: F.cross_entropy(tf.ones(0, 3), np.array([], dtype=int)),
        lambda: F.binary_cross_entropy_with_logits(tf.ones(2), tf.tensor([0, 2])),
        lambda: nn.Linear(0, 2),
        lambda: nn.Dropout(1),
        lambda: nn.LayerNorm(3, eps=0),
        lambda: nn.Sequential("not a layer"),
    ],
)
def test_validation(call):
    with pytest.raises((ValueError, TypeError)):
        call()


def test_layernorm_gradient(gradcheck):
    layer = nn.LayerNorm(3)
    x = np.array([[1.0, 3.0, 2.0], [-1.0, 0.0, 4.0]])
    gradcheck(layer, [x])
    result = layer(tf.tensor(x)).numpy()
    np.testing.assert_allclose(result.mean(axis=-1), 0, atol=1e-12)
    np.testing.assert_allclose(result.var(axis=-1), 1, atol=2e-5)
    layer.zero_grad()  # The preceding gradient check already accumulated gradients.
    layer(tf.tensor(x)).sum().backward()
    np.testing.assert_equal(layer.bias.grad, [2, 2, 2])


def test_embedding_repeated_indices():
    layer = nn.Embedding(4, 3)
    result = layer(np.array([1, 1, 3]))
    result.sum().backward()
    np.testing.assert_equal(layer.weight.grad, [[0, 0, 0], [2, 2, 2], [0, 0, 0], [1, 1, 1]])
    with pytest.raises(TypeError):
        layer(tf.tensor([1]))
    with pytest.raises(ValueError):
        layer(np.array([-1]))


def test_dropout_and_flatten():
    layer = nn.Dropout(0.5)
    x = tf.ones(10000, requires_grad=True)
    tf.manual_seed(3)
    result = layer(x)
    assert set(np.unique(result.numpy())) == {0.0, 2.0}
    assert result.mean().item() == pytest.approx(1, abs=0.04)
    result.sum().backward()
    np.testing.assert_equal(x.grad, result.numpy())
    layer.eval()
    assert layer(x) is x
    assert nn.Flatten()(tf.ones(2, 3, 4)).shape == (2, 12)


def test_conv2d_values_and_gradients(gradcheck):
    layer = nn.Conv2d(1, 1, 2, bias=False)
    layer.weight._assign(np.ones((1, 1, 2, 2)))
    x = np.arange(9.0).reshape(1, 1, 3, 3)
    np.testing.assert_equal(layer(tf.tensor(x)).numpy(), [[[[8, 12], [20, 24]]]])
    gradcheck(layer, [x])
    layer.zero_grad()
    layer(tf.tensor(x)).sum().backward()
    np.testing.assert_equal(layer.weight.grad, [[[[8, 12], [20, 24]]]])


def test_conv_stride_padding_parameter_gradients():
    tf.manual_seed(6)
    layer = nn.Conv2d(2, 2, 2, stride=2, padding=1)
    x = tf.randn(2, 2, 3, 4, requires_grad=True)
    result = (layer(x) ** 2).sum()
    result.backward()
    for parameter in layer.parameters():
        original = parameter.numpy()
        expected = np.empty_like(original)
        for index in np.ndindex(original.shape):
            plus, minus = original.copy(), original.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            parameter._assign(plus)
            yp = (layer(x) ** 2).sum().item()
            parameter._assign(minus)
            ym = (layer(x) ** 2).sum().item()
            expected[index] = (yp - ym) / (2e-6)
        parameter._assign(original)
        np.testing.assert_allclose(parameter.grad, expected, atol=1e-6, rtol=1e-5)


def test_pooling_overlap_and_ties(gradcheck):
    layer = nn.MaxPool2d(2, stride=1)
    a = np.array([[[[1.0, 2.0, 3.0], [4.0, 9.0, 6.0], [7.0, 8.0, 5.0]]]])
    gradcheck(layer, [a])
    x = tf.tensor(a, requires_grad=True)
    layer(x).sum().backward()
    expected = np.zeros_like(a)
    expected[0, 0, 1, 1] = 4
    np.testing.assert_equal(x.grad, expected)
    tied = tf.ones(1, 1, 2, 2, requires_grad=True)
    nn.MaxPool2d(2)(tied).sum().backward()
    np.testing.assert_equal(tied.grad, [[[[1, 0], [0, 0]]]])


def test_attention_composition(gradcheck):
    rng = np.random.default_rng(2)
    q, k, v = [rng.normal(size=(1, 3, 2)) for _ in range(3)]
    gradcheck(lambda q, k, v: ((q @ k.transpose(-2, -1)) / np.sqrt(2)).softmax() @ v, [q, k, v])
