import numpy as np
import pytest

import tensorforge as tf
from tensorforge import nn, optim
from tensorforge.data import DataLoader, TensorDataset


def test_sgd_momentum_and_decay():
    p = nn.Parameter([1.0, 2.0])
    optimizer = optim.SGD([p], lr=0.1, momentum=0.5, weight_decay=0.1)
    p.grad = np.array([2.0, -1.0])
    optimizer.step()
    np.testing.assert_allclose(p.numpy(), [0.79, 2.08])
    p.grad = np.array([1.0, 1.0])
    optimizer.step()
    np.testing.assert_allclose(p.numpy(), [0.5771, 1.9992])
    optimizer.zero_grad()
    before = p.numpy()
    optimizer.step()
    np.testing.assert_equal(p.numpy(), before)


def test_adam_reference_equations():
    p = nn.Parameter([1.0, -2.0])
    optimizer = optim.Adam([p], lr=0.02, betas=(0.8, 0.9), eps=1e-7)
    expected, m, v = p.numpy(), np.zeros(2), np.zeros(2)
    for step, g in enumerate(
        [np.array([2.0, -3.0]), np.array([-1.0, 4.0]), np.array([0.5, 0.2])], 1
    ):
        p.grad = g
        optimizer.step()
        m, v = 0.8 * m + 0.2 * g, 0.9 * v + 0.1 * g * g
        expected -= 0.02 * (m / (1 - 0.8**step)) / (np.sqrt(v / (1 - 0.9**step)) + 1e-7)
        np.testing.assert_allclose(p.numpy(), expected)


def test_optimizer_rejects_bad_gradients_before_updates():
    a, b = nn.Parameter([1.0]), nn.Parameter([2.0])
    optimizer = optim.SGD([a, b])
    a.grad, b.grad = np.array([1.0]), np.array([np.nan])
    with pytest.raises(ValueError):
        optimizer.step()
    assert a.item() == 1
    with pytest.raises(ValueError):
        optim.SGD([a, a])
    with pytest.raises(ValueError):
        optim.Adam([])
    with pytest.raises(TypeError):
        optim.SGD([a + 1])


def test_loader_alignment_seed_and_epoch():
    x = tf.tensor(np.arange(10).reshape(10, 1))
    labels = tf.tensor(np.arange(10), dtype=np.int64)
    dataset = TensorDataset(x, labels)
    first = DataLoader(dataset, batch_size=4, shuffle=True, seed=7)
    second = DataLoader(dataset, batch_size=4, shuffle=True, seed=7)
    assert len(first) == 3
    assert len(DataLoader(dataset, batch_size=4, drop_last=True)) == 2

    def collect(loader):
        indices = []
        for features, target in loader:
            np.testing.assert_equal(features.numpy()[:, 0], target.numpy())
            assert target.dtype == np.int64
            indices.extend(target.numpy().tolist())
        return indices

    a, b = collect(first), collect(second)
    assert a == b and sorted(a) == list(range(10))
    assert collect(first) != a
    assert list(DataLoader(TensorDataset(tf.zeros(0, 2)))) == []
    with pytest.raises(ValueError):
        DataLoader(dataset, batch_size=0)
    with pytest.raises(ValueError):
        TensorDataset(tf.ones(2), tf.ones(3))


@pytest.mark.parametrize("optimizer_class", [optim.SGD, optim.Adam])
def test_checkpoint_exact_next_update(tmp_path, optimizer_class):
    tf.manual_seed(4)
    model = nn.Linear(2, 1)
    kwargs = {"momentum": 0.8} if optimizer_class is optim.SGD else {}
    optimizer = optimizer_class(model.parameters(), lr=0.01, **kwargs)
    x, y = tf.tensor([[1.0, 2.0], [2.0, -1.0]]), tf.tensor([[3.0], [0.0]])

    def update(network, opt):
        opt.zero_grad()
        nn.MSELoss()(network(x), y).backward()
        opt.step()

    update(model, optimizer)
    path = tmp_path / "training.npz"
    tf.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 1}, path)
    state = tf.load(path)
    clone = nn.Linear(2, 1)
    other = optimizer_class(clone.parameters())
    clone.load_state_dict(state["model"])
    other.load_state_dict(state["optimizer"])
    update(model, optimizer)
    update(clone, other)
    for a, b in zip(model.parameters(), clone.parameters()):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    assert state["epoch"] == 1


def test_checkpoint_validation_is_atomic(tmp_path):
    model = nn.Linear(2, 1)
    before = model.state_dict()
    bad = model.state_dict()
    bad["weight"][:] = 100
    bad["bias"] = np.ones(9)
    with pytest.raises(ValueError):
        model.load_state_dict(bad)
    np.testing.assert_equal(model.weight.numpy(), before["weight"])
    with pytest.raises(ValueError):
        model.load_state_dict({})
    with pytest.raises(TypeError):
        tf.save({"unsupported": object()}, tmp_path / "bad.npz")
    assert not (tmp_path / "bad.npz").exists()


def test_adam_skipped_parameters_and_invalid_state():
    a, b = nn.Parameter([1.0]), nn.Parameter([2.0])
    optimizer = optim.Adam([a, b])
    a.grad = np.array([1.0])
    optimizer.step()
    assert optimizer.state_dict()["steps"] == [1, 0]
    state = optimizer.state_dict()
    state["v"][0][:] = -1
    with pytest.raises(ValueError):
        optimizer.load_state_dict(state)
    assert optimizer.state_dict()["v"][0][0] > 0
