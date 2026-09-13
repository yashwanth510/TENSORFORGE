"""Train a Transformer encoder to detect a token in padded sequences."""

import numpy as np

import tensorforge as tf
from tensorforge import nn, optim
from tensorforge.data import padding_mask


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(6, 8)
        self.position = nn.SinusoidalPositionEncoding(8, 8)
        self.encoder = nn.TransformerEncoder(8, 2, hidden_size=16)
        self.output = nn.Linear(8, 2)

    def forward(self, tokens, mask):
        x = self.encoder(self.position(self.embedding(tokens)), padding_mask=mask)
        keep = tf.tensor(~mask.numpy()).unsqueeze(-1)
        pooled = (x * keep).sum(axis=1) / keep.sum(axis=1)
        return self.output(pooled)


def main():
    tf.manual_seed(12)
    rng = np.random.default_rng(12)
    lengths = rng.integers(2, 6, size=48)
    mask = padding_mask(lengths, max_length=5)
    ids = rng.integers(1, 6, size=(48, 5))
    ids[mask.numpy()] = 0
    labels = (ids == 5).any(axis=1).astype(np.int64)
    tokens, target = tf.tensor(ids, dtype=np.int64), tf.tensor(labels, dtype=np.int64)
    model = Classifier()
    optimizer = optim.AdamW(model.parameters(), lr=0.015)
    initial = nn.CrossEntropyLoss()(model(tokens, mask), target).item()
    for _ in range(100):
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(tokens, mask), target)
        loss.backward()
        optimizer.step()
    model.eval()
    with tf.no_grad():
        logits = model(tokens, mask)
        final = nn.CrossEntropyLoss()(logits, target).item()
        accuracy = np.mean(logits.numpy().argmax(-1) == labels)
    print(f"Encoder classification: loss {initial:.6f} -> {final:.6f}, accuracy={accuracy:.0%}")
    assert accuracy >= 0.98 and final < 0.05


if __name__ == "__main__":
    main()
