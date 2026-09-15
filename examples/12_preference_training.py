"""Prefer one next-token response using DPO and low-rank fine-tuning."""

import numpy as np

import tensorforge as tf
from tensorforge.finetuning import apply_lora, dpo_loss, trainable_parameters
from tensorforge.models import CausalLanguageModel, LanguageModelConfig


def main():
    tf.manual_seed(63)
    model = CausalLanguageModel(
        LanguageModelConfig(8, features=8, num_heads=2, num_layers=1, hidden_size=16, max_length=8)
    )
    prompt = tf.tensor([[1, 4], [1, 5]], dtype=np.int64)
    with tf.no_grad():
        reference = model(prompt)[:, -1].log_softmax()
        ref_chosen, ref_rejected = reference[:, 6], reference[:, 7]
    apply_lora(model, rank=2, alpha=4)
    optimizer = tf.optim.Adam(trainable_parameters(model), lr=0.03)
    initial = (ref_chosen - ref_rejected).mean().item()
    for _ in range(100):
        optimizer.zero_grad()
        logp = model(prompt)[:, -1].log_softmax()
        loss = dpo_loss(logp[:, 6], logp[:, 7], ref_chosen, ref_rejected, beta=2)
        loss.backward()
        optimizer.step()
    with tf.no_grad():
        logp = model(prompt)[:, -1].log_softmax()
        final = (logp[:, 6] - logp[:, 7]).mean().item()
    print(f"Preferred-response log-probability margin: {initial:.5f} -> {final:.5f}")
    assert final > initial + 0.02
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


if __name__ == "__main__":
    main()
