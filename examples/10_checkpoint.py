"""Save model AND optimizer state; reproduce the next training update."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import tensorforge as tf
from tensorforge import nn, optim


def main():
    tf.manual_seed(10)
    model = nn.Linear(1, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    x, y = tf.tensor([[1.0], [2.0]]), tf.tensor([[3.0], [5.0]])

    def update(network, opt):
        opt.zero_grad()
        nn.MSELoss()(network(x), y).backward()
        opt.step()

    update(model, optimizer)
    # A temporary directory keeps this demonstration from leaving artifacts.
    # Use Path("checkpoints/run.npz") after creating checkpoints/ for real runs.
    with TemporaryDirectory() as directory:
        path = Path(directory) / "training.npz"
        tf.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 1}, path)
        restored = tf.load(path)
        # Architecture is Python code, so recreate it before loading arrays.
        clone = nn.Linear(1, 1)
        clone_optimizer = optim.Adam(clone.parameters())
        clone.load_state_dict(restored["model"])
        clone_optimizer.load_state_dict(restored["optimizer"])
        update(model, optimizer)
        update(clone, clone_optimizer)
        for original, copied in zip(model.parameters(), clone.parameters()):
            np.testing.assert_array_equal(original.numpy(), copied.numpy())
    print("Model and Adam state restored; the next update matches exactly.")


if __name__ == "__main__":
    main()
