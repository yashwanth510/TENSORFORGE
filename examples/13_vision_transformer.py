"""Train a small Vision Transformer on generated image patterns."""

import numpy as np

import tensorforge as tf
from tensorforge import nn, optim


def main():
    tf.manual_seed(74)
    rng = np.random.default_rng(74)
    labels = np.arange(24) % 2
    images = rng.normal(0, 0.02, (24, 1, 4, 4))
    images[labels == 0, 0, :, 1:3] += 1
    images[labels == 1, 0, 1:3, :] += 1
    x, y = tf.tensor(images), tf.tensor(labels, dtype=np.int64)
    model = nn.VisionTransformer(4, 2, 1, 2, features=8, num_heads=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(80):
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()
    with tf.evaluation.evaluating(model):
        accuracy = tf.evaluation.classification_accuracy(model(x), y)
    print(f"Vision Transformer training accuracy: {accuracy:.0%}")
    assert accuracy == 1


if __name__ == "__main__":
    main()
