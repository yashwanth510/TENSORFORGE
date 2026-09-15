"""Learn masked-token recovery with a bidirectional Transformer encoder."""

import numpy as np

import tensorforge as tf
from tensorforge import nn, optim


def main():
    tf.manual_seed(52)
    token = np.tile(np.arange(4, 8), 4)
    inputs = tf.tensor(np.stack((token, np.full_like(token, 3), token), axis=1), dtype=np.int64)
    labels = tf.tensor(
        np.stack((np.full_like(token, -100), token, np.full_like(token, -100)), axis=1),
        dtype=np.int64,
    )
    model = nn.MaskedLanguageModel(8, features=8, num_heads=2, num_layers=1, max_length=8)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    initial = nn.CrossEntropyLoss()(model(inputs), labels).item()
    for _ in range(300):
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(inputs), labels)
        loss.backward()
        optimizer.step()
    with tf.evaluation.evaluating(model):
        logits = model(inputs)
        final = nn.CrossEntropyLoss()(logits, labels).item()
        accuracy = np.mean(logits.numpy()[:, 1].argmax(-1) == token)
    print(f"Masked tokens: loss {initial:.5f} -> {final:.5f}, accuracy={accuracy:.0%}")
    assert final < 0.05 and accuracy == 1


if __name__ == "__main__":
    main()
