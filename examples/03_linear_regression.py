"""Learn y = 3*x + 2 from generated data; no external dataset required."""

import numpy as np

import tensorforge as tf
from tensorforge import nn, optim
from tensorforge.data import DataLoader, TensorDataset


def main():
    tf.manual_seed(7)
    x = tf.tensor(np.linspace(-1, 1, 80).reshape(-1, 1))
    y = 3 * x + 2
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True, seed=7)
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_function = nn.MSELoss()
    initial = loss_function(model(x), y).item()

    for epoch in range(80):
        for features, targets in loader:
            # Clear previous gradients; backward() accumulates by design.
            optimizer.zero_grad()
            prediction = model(features)
            loss = loss_function(prediction, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    with tf.no_grad():
        final = loss_function(model(x), y).item()
        prediction = model(tf.tensor([[4.0]])).item()
    print(f"MSE: {initial:.6f} -> {final:.10f}")
    print(f"weight={model.weight.item():.6f}, bias={model.bias.item():.6f}")
    print(f"Prediction for x=4: {prediction:.6f}; expected 14")
    assert final < 1e-8 and abs(prediction - 14) < 1e-3


if __name__ == "__main__":
    main()
