"""Train RNN, LSTM, and GRU models to average a short sequence."""

import numpy as np

import tensorforge as tf
from tensorforge import nn, optim


class SequenceModel(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.recurrent = layer(1, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        sequence, _ = self.recurrent(x)
        return self.output(sequence[:, -1])


def main():
    rng = np.random.default_rng(4)
    x = tf.tensor(rng.uniform(-1, 1, (24, 4, 1)))
    target = x.mean(axis=1)
    for layer in (nn.RNN, nn.LSTM, nn.GRU):
        tf.manual_seed(4)
        model = SequenceModel(layer)
        optimizer = optim.Adam(model.parameters(), lr=0.02)
        initial = nn.MSELoss()(model(x), target).item()
        for _ in range(160):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(x), target)
            loss.backward()
            optimizer.step()
        model.eval()
        with tf.no_grad():
            final = nn.MSELoss()(model(x), target).item()
        print(f"{layer.__name__}: MSE {initial:.6f} -> {final:.6f}")
        assert final < 0.005 and final < initial * 0.2


if __name__ == "__main__":
    main()
