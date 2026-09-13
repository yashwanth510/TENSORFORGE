"""Train a tiny CNN to distinguish synthetic vertical and horizontal bars."""

import numpy as np

import tensorforge as tf
from tensorforge import nn, optim


def main():
    tf.manual_seed(11)
    rng = np.random.default_rng(11)
    images = rng.normal(0, 0.05, (32, 1, 6, 6))
    labels = np.arange(32) % 2
    images[labels == 0, 0, :, 2:4] += 1  # Vertical bars.
    images[labels == 1, 0, 2:4, :] += 1  # Horizontal bars.
    x, y = tf.tensor(images), tf.tensor(labels, dtype=np.int64)
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(36, 2)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    for _ in range(60):
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    with tf.no_grad():
        logits = model(x)
        accuracy = np.mean(logits.numpy().argmax(axis=1) == labels)
        loss = nn.CrossEntropyLoss()(logits, y).item()
    print(f"Synthetic image training accuracy={accuracy:.0%}, loss={loss:.6f}")
    print("This checks learning on training data; it is not a generalization benchmark.")
    assert accuracy == 1.0 and loss < 0.02


if __name__ == "__main__":
    main()
