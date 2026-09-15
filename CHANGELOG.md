# Changelog

## 0.2.0 — 2026-09-15

- Added byte-pair tokenization, masked/span corruption, and paired/instruction data.
- Added masked language models, Vision Transformers, multimodal classifiers, and experts.
- Added LoRA, adapters, preference losses, native model bundles, and evaluation.
- Added tiled CPU attention, position biases, beam search, greedy speculative decoding,
  paged CPU cache storage, activation checkpointing, and int8 weight storage.
- Added local process-based data-parallel gradients and document retrieval.
- Added tests and runnable training examples. GPU, multi-node training, third-party
  pretrained compatibility, and a complete PPO/RLHF pipeline remain unsupported.

## 0.1.0 — 2026-09-13

Initial CPU release:

- Tensors, scalar autograd, and first-order tensor autograd.
- Dense, convolutional, recurrent, and Transformer networks.
- Attention masks, rotary positions, grouped-query heads, and cached generation.
- GPT-style and modern decoder language-model configurations.
- Losses, SGD, Adam, AdamW, clipping, warmup, and learning-rate schedules.
- Batching, padding, character tokenization, and resumable training checkpoints.
- Runnable examples, automated tests, source/wheel packaging, and GitHub CI.

See README for supported APIs and limitations.
