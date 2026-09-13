import numpy as np
import pytest

import tensorforge as tf
from tensorforge import nn, optim
from tensorforge.data import DataLoader, TensorDataset, pad_sequence, padding_mask
from tensorforge.text import CharacterTokenizer, TextDataset
from tensorforge.training import LRScheduler, clip_grad_norm_, load_checkpoint, save_checkpoint


def test_adamw_decoupled_decay():
    p = nn.Parameter([2.0, -3.0])
    optimizer = optim.AdamW([p], lr=0.1, weight_decay=0.2)
    p.grad = np.zeros(2)
    optimizer.step()
    np.testing.assert_allclose(p.numpy(), [1.96, -2.94])
    p.grad = np.array([1.0, -1.0])
    optimizer.step()
    m = np.array([0.1, -0.1]) / (1 - 0.9**2)
    v = np.array([0.001, 0.001]) / (1 - 0.999**2)
    expected = np.array([1.96, -2.94]) * 0.98 - 0.1 * m / (np.sqrt(v) + 1e-8)
    np.testing.assert_allclose(p.numpy(), expected)
    other = optim.AdamW([nn.Parameter([1.0, 1.0])])
    other.load_state_dict(optimizer.state_dict())
    assert other.state_dict()["type"] == "AdamW"
    np.testing.assert_equal(other.state_dict()["m"], optimizer.state_dict()["m"])


def test_clip_norm_and_accumulation():
    p, q = nn.Parameter([0.0]), nn.Parameter([0.0])
    p.grad, q.grad = np.array([3.0]), np.array([4.0])
    assert clip_grad_norm_([p, q], 2) == 5
    np.testing.assert_allclose([p.grad[0], q.grad[0]], [1.2, 1.6])
    p.grad = np.array([np.inf])
    with pytest.raises(ValueError):
        clip_grad_norm_([p], 1)
    p.grad = np.array([1e308])
    clip_grad_norm_([p], 1)
    np.testing.assert_allclose(p.grad, [1.0])
    tf.manual_seed(22)
    model = nn.Linear(2, 1)
    x, y = tf.randn(4, 2), tf.randn(4, 1)
    nn.MSELoss()(model(x), y).backward()
    reference = [p.grad.copy() for p in model.parameters()]
    model.zero_grad()
    for start in (0, 2):
        (nn.MSELoss()(model(x[start : start + 2]), y[start : start + 2]) / 2).backward()
    for p, expected in zip(model.parameters(), reference):
        np.testing.assert_allclose(p.grad, expected)


@pytest.mark.parametrize("schedule", ["constant", "step", "cosine"])
def test_schedules_and_restore(schedule):
    optimizer = optim.SGD([nn.Parameter([1.0])], lr=0.1)
    scheduler = LRScheduler(
        optimizer,
        schedule=schedule,
        warmup_steps=2,
        total_steps=6,
        step_size=2,
        gamma=0.5,
        min_lr=0.01,
    )
    rates = [optimizer.lr]
    for _ in range(6):
        rates.append(scheduler.step())
    assert rates[:3] == pytest.approx([0.05, 0.1, 0.1])
    if schedule == "step":
        assert rates[4] == pytest.approx(0.05)
    elif schedule == "cosine":
        assert rates[-1] == pytest.approx(0.01)
    else:
        assert rates[-1] == pytest.approx(0.1)
    state = scheduler.state_dict()
    other = LRScheduler(optim.SGD([nn.Parameter([1.0])], lr=0.8))
    other.load_state_dict(state)
    assert other.step() == pytest.approx(scheduler.step())


def test_padding_custom_collate_and_text():
    sequences = [tf.tensor([1, 2], dtype=np.int64), tf.tensor([3], dtype=np.int64)]
    padded = pad_sequence(sequences, padding_value=0)
    np.testing.assert_equal(padded.numpy(), [[1, 2], [3, 0]])
    np.testing.assert_equal(padding_mask([2, 1]).numpy(), [[False, False], [False, True]])
    loader = DataLoader(sequences, batch_size=2, collate_fn=lambda samples: pad_sequence(samples))
    np.testing.assert_equal(next(iter(loader)).numpy(), padded.numpy())
    tokenizer = CharacterTokenizer("hello\n")
    encoded = tokenizer.encode("hello", add_bos=True, add_eos=True)
    assert encoded[0] == tokenizer.bos_id and encoded[-1] == tokenizer.eos_id
    assert tokenizer.decode(encoded) == "hello"
    assert tokenizer.encode("?") == [tokenizer.unk_id]
    clone = CharacterTokenizer.from_state_dict(tokenizer.state_dict())
    assert clone.encode("hello") == tokenizer.encode("hello")
    dataset = TextDataset(tokenizer.encode("hellohello"), sequence_length=4, stride=2)
    assert len(dataset) == 3
    x, y = dataset[1]
    np.testing.assert_equal(x.numpy()[1:], y.numpy()[:-1])
    with pytest.raises(ValueError):
        CharacterTokenizer.from_state_dict({"type": "character", "tokens": ["bad"]})


def test_loader_resume_mid_epoch():
    dataset = TensorDataset(tf.arange(9))
    loader = DataLoader(dataset, batch_size=2, shuffle=True, seed=11)
    next(iter(loader))
    state = loader.state_dict()
    expected = [batch[0].numpy() for batch in loader]
    other = DataLoader(dataset, batch_size=2, shuffle=True, seed=99)
    other.load_state_dict(state)
    actual = [batch[0].numpy() for batch in other]
    for a, b in zip(actual, expected):
        np.testing.assert_equal(a, b)
    assert len(actual) == 4
    # The following epoch also matches because the shuffle RNG was restored.
    np.testing.assert_equal(next(iter(loader))[0].numpy(), next(iter(other))[0].numpy())


def test_complete_stochastic_checkpoint(tmp_path):
    tf.manual_seed(8)
    model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Dropout(0.4), nn.Linear(4, 1))
    optimizer = optim.AdamW(model.parameters(), lr=0.02)
    scheduler = LRScheduler(optimizer, "cosine", total_steps=10)
    dataset = TensorDataset(tf.randn(8, 2), tf.randn(8, 1))
    loader = DataLoader(dataset, batch_size=2, shuffle=True, seed=8)

    def update(network, opt, sched, batch):
        x, y = batch
        opt.zero_grad()
        loss = nn.MSELoss()(network(x), y)
        loss.backward()
        opt.step()
        sched.step()
        return loss.item()

    update(model, optimizer, scheduler, next(iter(loader)))
    path = tmp_path / "run.npz"
    save_checkpoint(path, model, optimizer, scheduler, loader, step=1, extra={"note": "test"})
    expected_loss = update(model, optimizer, scheduler, next(loader))
    expected = model.state_dict()
    clone = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Dropout(0.4), nn.Linear(4, 1))
    other_optimizer = optim.AdamW(clone.parameters())
    other_scheduler = LRScheduler(other_optimizer)
    other_loader = DataLoader(dataset, batch_size=2, shuffle=True, seed=100)
    state = load_checkpoint(path, clone, other_optimizer, other_scheduler, other_loader)
    actual_loss = update(clone, other_optimizer, other_scheduler, next(iter(other_loader)))
    assert actual_loss == expected_loss
    assert state == {"step": 1, "extra": {"note": "test"}}
    for name, value in clone.state_dict().items():
        np.testing.assert_array_equal(value, expected[name])


def test_checkpoint_failed_restore_leaves_model_unchanged(tmp_path):
    model = nn.Linear(1, 1)
    optimizer = optim.AdamW(model.parameters())
    path = tmp_path / "state.npz"
    save_checkpoint(path, model, optimizer)
    state = tf.load(path)
    state["model"]["weight"][:] = 100
    state["optimizer"]["steps"] = [-1, -1]
    tf.save(state, path)
    original = model.state_dict()
    with pytest.raises(ValueError):
        load_checkpoint(path, model, optimizer)
    np.testing.assert_equal(model.weight.numpy(), original["weight"])
