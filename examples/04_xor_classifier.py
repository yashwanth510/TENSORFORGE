"""A nonlinear classifier: XOR is true when two bits differ."""

import numpy as np

import tensorforge as tf
from tensorforge import nn, optim


def main():
    tf.manual_seed(42)
    x = tf.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Explicit integer dtype: class IDs are indices, not continuous numbers.
    target = tf.tensor([0, 1, 1, 0], dtype=np.int64)
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    loss_function = nn.CrossEntropyLoss()
    initial = loss_function(model(x), target).item()

    for step in range(500):
        optimizer.zero_grad()
        loss = loss_function(model(x), target)
        loss.backward()
        optimizer.step()

    model.eval()
    with tf.no_grad():
        logits = model(x)
        final = loss_function(logits, target).item()
        predictions = logits.numpy().argmax(axis=1)
    accuracy = np.mean(predictions == target.numpy())
    print(f"Cross-entropy: {initial:.6f} -> {final:.6f}")
    print(f"Predictions: {predictions.tolist()}, accuracy={accuracy:.0%}")
    assert accuracy == 1.0 and final < 0.02


if __name__ == "__main__":
    main()
