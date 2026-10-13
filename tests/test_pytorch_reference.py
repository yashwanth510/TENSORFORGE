"""Independent CPU float64 references; PyTorch is an optional test dependency.

Copy identical parameters into native PyTorch layers, then compare outputs,
losses, input/state gradients, and every corresponding parameter gradient.
"""

from functools import partial

import numpy as np
import pytest

import tensorforge as tf
from tensorforge import nn
from tensorforge.models import CausalLanguageModel, LanguageModelConfig

torch = pytest.importorskip("torch", reason="Install the reference extra for PyTorch comparisons")
F = torch.nn.functional


def close(actual, expected):
    if isinstance(actual, tf.Tensor):
        actual = actual.numpy()
    if isinstance(expected, torch.Tensor):
        expected = expected.detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-8)


def inputs(shape, seed=7):
    values = np.random.default_rng(seed).normal(size=shape)
    return tf.tensor(values, requires_grad=True), torch.tensor(values, requires_grad=True)


def copy_pairs(pairs):
    with torch.no_grad():
        for actual, reference in pairs:
            reference.copy_(torch.tensor(actual.numpy()))


def compare_backward(actual, reference, input_pairs=(), parameter_pairs=()):
    close(actual, reference)
    # Scalar losses use ordinary backward; vector outputs exercise a nonuniform VJP.
    upstream = (
        np.ones(actual.shape)
        if actual.size == 1
        else np.linspace(-0.7, 1.3, actual.size).reshape(actual.shape)
    )
    actual.backward(upstream)
    reference.backward(torch.tensor(upstream))
    for left, right in (*input_pairs, *parameter_pairs):
        assert left.grad is not None and right.grad is not None
        close(left.grad, right.grad)


def linear_pairs(actual, reference):
    pairs = [(actual.weight, reference.weight)]
    if actual.bias is not None:
        pairs.append((actual.bias, reference.bias))
    return pairs


def norm_pairs(actual, reference):
    return [(actual.weight, reference.weight), (actual.bias, reference.bias)]


def attention_pairs(actual, reference):
    # PyTorch packs Q/K/V into one parameter. Slices copy the forward weights;
    # gradient comparisons below use the original packed parameter's gradient.
    width = actual.embed_dim
    with torch.no_grad():
        for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
            projection = getattr(actual, name)
            reference.in_proj_weight[index * width : (index + 1) * width].copy_(
                torch.tensor(projection.weight.numpy())
            )
            reference.in_proj_bias[index * width : (index + 1) * width].copy_(
                torch.tensor(projection.bias.numpy())
            )
    pairs = linear_pairs(actual.out_proj, reference.out_proj)
    copy_pairs(pairs)
    return pairs


def compare_attention_gradients(actual, reference):
    width = actual.embed_dim
    for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
        projection = getattr(actual, name)
        close(
            projection.weight.grad,
            reference.in_proj_weight.grad[index * width : (index + 1) * width],
        )
        close(
            projection.bias.grad, reference.in_proj_bias.grad[index * width : (index + 1) * width]
        )


@pytest.fixture(autouse=True)
def deterministic_initialization():
    tf.manual_seed(23)


def test_broadcast_matmul_and_reduction():
    x, rx = inputs((2, 1, 3, 4))
    weight, rw = inputs((1, 2, 4, 5), seed=8)
    bias, rb = inputs((5,), seed=9)
    result = ((x @ weight + bias).tanh() ** 2).mean(axis=(0, 2))
    reference = ((rx @ rw + rb).tanh() ** 2).mean(dim=(0, 2))
    compare_backward(result, reference, [(x, rx), (weight, rw), (bias, rb)])


@pytest.mark.parametrize(
    "activation", ["relu", "sigmoid", "tanh", "gelu", "silu", "softmax", "log_softmax"]
)
def test_activations(activation):
    x, rx = inputs((2, 3, 4))
    if activation == "gelu":
        reference = F.gelu(rx, approximate="tanh")
    elif activation in ("softmax", "log_softmax"):
        reference = getattr(F, activation)(rx, dim=-1)
    else:
        reference = getattr(F, activation)(rx)
    compare_backward(getattr(x, activation)(), reference, [(x, rx)])


def test_linear_layernorm_network():
    model = nn.Sequential(nn.Linear(3, 5), nn.LayerNorm(5), nn.Tanh(), nn.Linear(5, 2))
    reference = torch.nn.Sequential(
        torch.nn.Linear(3, 5), torch.nn.LayerNorm(5), torch.nn.Tanh(), torch.nn.Linear(5, 2)
    ).double()
    pairs = linear_pairs(model[0], reference[0]) + norm_pairs(model[1], reference[1])
    pairs += linear_pairs(model[3], reference[3])
    copy_pairs(pairs)
    x, rx = inputs((2, 4, 3))
    compare_backward(model(x), reference(rx), [(x, rx)], pairs)


def test_embedding_repeated_ids():
    model, reference = nn.Embedding(7, 4), torch.nn.Embedding(7, 4).double()
    pairs = [(model.weight, reference.weight)]
    copy_pairs(pairs)
    ids = np.array([[1, 1, 3], [3, 6, 1]])
    compare_backward(model(ids), reference(torch.tensor(ids)), parameter_pairs=pairs)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("smoothing", [0.0, 0.2])
def test_cross_entropy_extreme_logits_and_ignored_targets(reduction, smoothing):
    values = np.array([[1000.0, -1000.0, 0.0], [-800.0, 800.0, 1.0], [3.0, 2.0, -1.0]])
    x, rx = tf.tensor(values, requires_grad=True), torch.tensor(values, requires_grad=True)
    targets = np.array([1, -100, 0])
    actual = nn.functional.cross_entropy(x, targets, reduction=reduction, label_smoothing=smoothing)
    expected = F.cross_entropy(
        rx, torch.tensor(targets), reduction=reduction, label_smoothing=smoothing
    )
    compare_backward(actual, expected, [(x, rx)])


def test_binary_cross_entropy_extreme_logits():
    values = np.array([-1000.0, -2.0, 0.0, 3.0, 1000.0])
    targets = np.array([1.0, 0.2, 0.5, 0.8, 0.0])
    x, rx = tf.tensor(values, requires_grad=True), torch.tensor(values, requires_grad=True)
    compare_backward(
        nn.functional.binary_cross_entropy_with_logits(x, tf.tensor(targets)),
        F.binary_cross_entropy_with_logits(rx, torch.tensor(targets)),
        [(x, rx)],
    )


@pytest.mark.parametrize("pool", ["max", "average"])
def test_convolution_pooling_network(pool):
    convolution, reference = (
        nn.Conv2d(2, 3, 3, stride=2, padding=1),
        torch.nn.Conv2d(2, 3, 3, stride=2, padding=1).double(),
    )
    pairs = linear_pairs(convolution, reference)
    copy_pairs(pairs)
    x, rx = inputs((2, 2, 7, 7))
    pooling = nn.MaxPool2d(2, stride=1) if pool == "max" else nn.AvgPool2d(2, stride=1)
    operation = F.max_pool2d if pool == "max" else F.avg_pool2d
    compare_backward(
        pooling(convolution(x)), operation(reference(rx), 2, stride=1), [(x, rx)], pairs
    )


@pytest.mark.parametrize("kind", ["RNN", "LSTM", "GRU"])
def test_recurrent_sequence_and_initial_states(kind):
    model = getattr(nn, kind)(3, 4)
    reference = getattr(torch.nn, kind)(3, 4, batch_first=True).double()
    pairs = [
        (model.input.weight, reference.weight_ih_l0),
        (model.input.bias, reference.bias_ih_l0),
        (model.hidden.weight, reference.weight_hh_l0),
        (model.hidden.bias, reference.bias_hh_l0),
    ]
    copy_pairs(pairs)
    x, rx = inputs((2, 5, 3))
    h, rh = inputs((2, 4), seed=11)
    input_pairs = [(x, rx), (h, rh)]
    if kind == "LSTM":
        c, rc = inputs((2, 4), seed=12)
        output, (final_h, final_c) = model(x, (h, c))
        expected, (expected_h, expected_c) = reference(rx, (rh.unsqueeze(0), rc.unsqueeze(0)))
        close(final_c, expected_c[0])
        result = tf.cat((output.flatten(), final_h.flatten(), final_c.flatten()))
        target = torch.cat((expected.flatten(), expected_h.flatten(), expected_c.flatten()))
        input_pairs.append((c, rc))
    else:
        output, final_h = model(x, h)
        expected, expected_h = reference(rx, rh.unsqueeze(0))
        result = tf.cat((output.flatten(), final_h.flatten()))
        target = torch.cat((expected.flatten(), expected_h.flatten()))
    close(output, expected)
    close(final_h, expected_h[0])
    compare_backward(result, target, input_pairs, pairs)


@pytest.mark.parametrize("mode", ["self", "causal", "cross"])
def test_native_multihead_attention(mode):
    model, reference = (
        nn.MultiheadAttention(8, 2),
        torch.nn.MultiheadAttention(8, 2, dropout=0, batch_first=True).double(),
    )
    pairs = attention_pairs(model, reference)
    x, rx = inputs((2, 3, 8))
    input_pairs = [(x, rx)]
    if mode == "cross":
        memory, rm = inputs((2, 4, 8), seed=13)
        input_pairs.append((memory, rm))
        padding = np.array([[False, False, True, False], [False, True, False, False]])
        result = model(x, memory, memory, key_padding_mask=padding)
        expected = reference(
            rx, rm, rm, key_padding_mask=torch.tensor(padding), need_weights=False
        )[0]
    else:
        mask = np.triu(np.ones((3, 3), dtype=bool), 1) if mode == "causal" else None
        result = model(x, is_causal=mode == "causal")
        expected = reference(
            rx, rx, rx, attn_mask=None if mask is None else torch.tensor(mask), need_weights=False
        )[0]
    compare_backward(result, expected, input_pairs, pairs)
    compare_attention_gradients(model, reference)


def encoder_pairs(model, reference):
    pairs = attention_pairs(model.attention, reference.self_attn)
    pairs += norm_pairs(model.norm1, reference.norm1) + norm_pairs(model.norm2, reference.norm2)
    pairs += linear_pairs(model.feed_forward.up, reference.linear1)
    pairs += linear_pairs(model.feed_forward.down, reference.linear2)
    copy_pairs(pairs)
    return pairs


def torch_encoder():
    return torch.nn.TransformerEncoderLayer(
        8,
        2,
        dim_feedforward=12,
        dropout=0,
        batch_first=True,
        norm_first=True,
        activation=partial(F.gelu, approximate="tanh"),
    ).double()


def test_transformer_encoder_block():
    model, reference = nn.TransformerEncoderLayer(8, 2, hidden_size=12), torch_encoder()
    pairs = encoder_pairs(model, reference)
    x, rx = inputs((2, 4, 8))
    padding = np.array([[False, False, False, True], [False, False, True, True]])
    compare_backward(
        model(x, padding_mask=padding),
        reference(rx, src_key_padding_mask=torch.tensor(padding)),
        [(x, rx)],
        pairs,
    )
    compare_attention_gradients(model.attention, reference.self_attn)


class TorchGPT(torch.nn.Module):
    """Equivalent GPT recipe assembled from native PyTorch components."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 8)
        self.position = torch.nn.Embedding(8, 8)
        self.layers = torch.nn.ModuleList([torch_encoder() for _ in range(2)])
        self.norm = torch.nn.LayerNorm(8)
        self.double()

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.embedding(tokens) + self.position(torch.arange(length))
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.norm(x) @ self.embedding.weight.T


def test_gpt_logits_loss_and_all_parameter_gradients():
    model = CausalLanguageModel(
        LanguageModelConfig(11, features=8, num_heads=2, num_layers=2, hidden_size=12, max_length=8)
    )
    reference = TorchGPT()
    pairs = [
        (model.embedding.weight, reference.embedding.weight),
        (model.position.embedding.weight, reference.position.weight),
    ]
    pairs += norm_pairs(model.norm, reference.norm)
    for block, target in zip(model.layers, reference.layers):
        pairs += encoder_pairs(block, target)
    copy_pairs(pairs)
    ids = np.array([[1, 4, 4, 6], [2, 3, 5, 7]])
    targets = np.array([[4, 4, 6, 2], [3, 5, 7, -100]])
    logits, expected = model(ids), reference(torch.tensor(ids))
    close(logits, expected)
    loss = nn.functional.cross_entropy(logits, targets, label_smoothing=0.1)
    reference_loss = F.cross_entropy(
        expected.reshape(-1, 11), torch.tensor(targets).flatten(), label_smoothing=0.1
    )
    compare_backward(loss, reference_loss, parameter_pairs=pairs)
    for block, target in zip(model.layers, reference.layers):
        compare_attention_gradients(block.attention, target.self_attn)


@pytest.mark.parametrize("kind", ["SGD", "Adam", "AdamW"])
def test_network_loss_gradients_and_optimizer_updates(kind):
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
    reference = torch.nn.Sequential(
        torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2)
    ).double()
    pairs = linear_pairs(model[0], reference[0]) + linear_pairs(model[2], reference[2])
    copy_pairs(pairs)
    options = dict(lr=0.02, weight_decay=0.03)
    options.update(dict(momentum=0.7) if kind == "SGD" else dict(betas=(0.8, 0.95), eps=1e-8))
    optimizer = getattr(tf.optim, kind)(model.parameters(), **options)
    expected_optimizer = getattr(torch.optim, kind)(reference.parameters(), **options)
    rng = np.random.default_rng(31)
    for _ in range(4):
        values, targets = rng.normal(size=(4, 3)), rng.normal(size=(4, 2))
        x, rx = tf.tensor(values, requires_grad=True), torch.tensor(values, requires_grad=True)
        optimizer.zero_grad()
        expected_optimizer.zero_grad()
        output, expected = model(x), reference(rx)
        close(output, expected)
        loss = nn.functional.mse_loss(output, tf.tensor(targets))
        expected_loss = F.mse_loss(expected, torch.tensor(targets))
        compare_backward(loss, expected_loss, [(x, rx)], pairs)
        optimizer.step()
        expected_optimizer.step()
        for actual, target in pairs:
            close(actual, target)
