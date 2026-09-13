"""Train an encoder-decoder Transformer, then copy sequences autoregressively."""

from itertools import product

import numpy as np

import tensorforge as tf
from tensorforge import nn, optim


class CopyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5, 12)
        self.position = nn.LearnedPositionEncoding(12, 8)
        self.transformer = nn.Transformer(12, 3, hidden_size=24)
        self.output = nn.Linear(12, 5)

    def forward(self, source, target):
        source = self.position(self.embedding(source))
        target = self.position(self.embedding(target))
        return self.output(self.transformer(source, target))


def main():
    tf.manual_seed(15)
    source = tf.tensor(list(product(range(1, 5), repeat=3)), dtype=np.int64)
    # Shift the target right; token 0 marks the beginning, never the answer.
    target_input = tf.cat((tf.zeros(len(source), 1, dtype=np.int64), source[:, :-1]), axis=1)
    model = CopyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.015)
    initial = nn.CrossEntropyLoss()(model(source, target_input), source).item()
    for _ in range(180):
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(source, target_input), source)
        loss.backward()
        tf.training.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with tf.no_grad():
        final = nn.CrossEntropyLoss()(model(source, target_input), source).item()
        generated = tf.zeros(len(source), 1, dtype=np.int64)
        for _ in range(source.shape[1]):
            next_ids = model(source, generated)[:, -1].argmax(axis=-1).unsqueeze(1)
            generated = tf.cat((generated, next_ids), axis=1)
        accuracy = np.mean(generated.numpy()[:, 1:] == source.numpy())
    print(
        f"Sequence copy: loss {initial:.6f} -> {final:.6f}, generated-token accuracy={accuracy:.0%}"
    )
    print("Source:", source.numpy()[5].tolist(), "Generated:", generated.numpy()[5, 1:].tolist())
    assert final < 0.05 and accuracy >= 0.98


if __name__ == "__main__":
    main()
