# TensorForge

[![Tests](https://github.com/yashwanth510/TENSORFORGE/actions/workflows/tests.yml/badge.svg)](https://github.com/yashwanth510/TENSORFORGE/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Tensors, automatic differentiation, and neural networks in Python.**

TensorForge is a CPU deep-learning library covering dense networks,
convolutional networks, recurrent networks, and Transformer models. It includes
training utilities, text generation, and resumable checkpoints in a small,
readable codebase.

NumPy handles array storage and computation. TensorForge implements the
computation graph, derivatives, layers, optimizers, and model architectures.
The library does not depend on PyTorch, TensorFlow, or JAX.

[Installation](#installation) · [Quick start](#quick-start) ·
[Transformers](#transformers) · [Examples](#examples) ·
[API](#api-reference) · [Testing](#testing-and-packaging) ·
[Limitations](#scope-and-limitations)

## Features

| Component | Included |
| --- | --- |
| Tensors | Broadcasting, indexing, matrix multiplication, reductions, shape operations, concatenation, stacking, masking |
| Autograd | First-order reverse-mode differentiation, gradient accumulation, detach, gradient-free inference |
| Networks | Dense layers, convolution, pooling, RNN, LSTM, GRU, embeddings, dropout, normalization |
| Transformers | Encoder, decoder, encoder–decoder, self/cross-attention, grouped-query attention, causal/padding masks |
| Position and feed-forward layers | Sinusoidal/learned positions, RoPE, GELU, SwiGLU, LayerNorm, RMSNorm |
| Language models | GPT-style and modern decoder configurations, tied embeddings, cached generation |
| Training | Losses, SGD, Adam, AdamW, clipping, warmup, step/cosine schedules, batching |
| Storage | Model/buffer state, optimizer/scheduler state, random state, loader progress, tokenizer/config metadata |

Version **0.1.0** supports small models and CPU experimentation. It is an
independent implementation with a limited API, not a drop-in PyTorch replacement.

## Installation

Python **3.10+** is required. From a new terminal:

```bash
git clone https://github.com/yashwanth510/TENSORFORGE.git
cd TENSORFORGE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For an existing checkout, activate its environment and run the install command.
On Windows PowerShell, activation is `.venv\Scripts\Activate.ps1`.
To install runtime dependencies only, use `python -m pip install .`.

- Runtime: `numpy>=1.24,<3`.
- Development: pytest, build, and Ruff, as declared in `pyproject.toml`.
- `-e` installs the source in editable mode.
- No GPU toolchain or downloaded dataset is required.

This project's package name is `tensorforge`. It is distributed through this
repository and locally built packages; **this release is not published to
PyPI**. A similarly named package elsewhere may be unrelated.

## Quick start

### Differentiate a tensor expression

```python
import tensorforge as tf

x = tf.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x * x).sum()
loss.backward()

print(loss.item())  # 14.0
print(x.grad)  # [2. 4. 6.]
```

### Train a neural network

```python
import tensorforge as tf
from tensorforge import nn, optim

x = tf.tensor([[0.0], [1.0], [2.0]])
y = tf.tensor([[1.0], [3.0], [5.0]])

tf.manual_seed(7)
model = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
optimizer = optim.Adam(model.parameters(), lr=0.02)
criterion = nn.MSELoss()

for step in range(300):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

model.eval()
with tf.no_grad():
    print(model(x).numpy())
```

`eval()` changes layer behavior, such as dropout and BatchNorm. `no_grad()`
stops recording the graph. Use both for inference.

### Define a custom module

```python
from tensorforge import nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(4, 16)
        self.output = nn.Linear(16, 3)

    def forward(self, x):
        return self.output(self.features(x).relu())
```

Parameters and child modules are discovered automatically, including modules
inside lists, tuples, `ModuleList`, `ModuleDict`, and ordinary dictionaries.
Shared parameters are returned once. Use `register_buffer(name, tensor)` for
persistent non-trainable state.

## Transformers

Transformer inputs are **batch first**: `(batch, time, features)`.
All TensorForge boolean attention masks use **True to block a position**.

```python
import tensorforge as tf
from tensorforge import nn

x = tf.randn(2, 6, 16)
encoder = nn.TransformerEncoder(features=16, num_heads=4, num_layers=2)
encoded = encoder(x)

attention = nn.MultiheadAttention(embed_dim=16, num_heads=4)
causal_output = attention(x, is_causal=True)

# Cross-attention: query sequence attends to a separate memory sequence.
memory = tf.randn(2, 9, 16)
cross_output = attention(x, memory, memory)
```

General attention masks broadcast to `(batch, heads, query_length, key_length)`.
`key_padding_mask` has shape `(batch, key_length)`. The functional attention
operation gives zero output for fully blocked rows. A module's final output
projection may add its bias afterward.

### Causal language models

```python
import numpy as np
import tensorforge as tf
from tensorforge.models import LanguageModelConfig, CausalLanguageModel

config = LanguageModelConfig(
    vocab_size=64,
    features=32,
    num_heads=4,
    num_layers=2,
    hidden_size=64,
    max_length=128,
    variant="modern",
    num_kv_heads=2,
)
model = CausalLanguageModel(config)
tokens = tf.tensor([[1, 5, 8, 3]], dtype=np.int64)
logits = model(tokens)  # (1, 4, 64)

# For useful text, train first; a freshly initialized model has random weights.
generated = model.generate(
    tokens,
    max_new_tokens=8,
    temperature=0.8,
    top_k=20,
    top_p=0.9,
    seed=7,
)
```

| Configuration | Position handling | Normalization | Feed-forward | Attention |
| --- | --- | --- | --- | --- |
| `variant="gpt"` | Learned embeddings | LayerNorm | GELU | Multi-head or grouped-query |
| `variant="modern"` | Rotary embeddings | RMSNorm | SwiGLU | Multi-head or grouped-query |

Both configurations use pre-normalized residual blocks and tied input/output
embeddings. The modern option implements common modern decoder building
blocks; it does not load pretrained Llama weights or claim model compatibility.

Generation supports greedy decoding (`temperature=0`), temperature sampling,
top-k, top-p, an optional end token, and a key/value cache. Prompts must be
nonempty, unpadded integer batches. Prompt plus requested output must fit
`max_length`. Generation restores the model's previous training modes.

For direct cache use, enter `no_grad()` after `eval()`:

```python
model.eval()
with tf.no_grad():
    prefix_logits, cache = model(tokens, use_cache=True)
    next_token = tf.tensor([[4]], dtype=np.int64)
    next_logits, cache = model(next_token, cache=cache, use_cache=True)
```

Caches store already projected keys and values. Do not reuse a cache after
changing model weights, its prompt, or its padding layout.

### Train on text and generate

```bash
# Train using the built-in text sample, save state, and generate.
python examples/09_language_model.py --checkpoint checkpoints/gpt.npz

# Train the modern decoder configuration.
python examples/09_language_model.py --variant modern --checkpoint checkpoints/modern.npz

# Train on your own UTF-8 file.
python examples/09_language_model.py --text-file /path/to/text.txt --steps 500 --checkpoint checkpoints/custom.npz

# Load the saved model and tokenizer for generation without retraining.
python examples/09_language_model.py --load checkpoints/modern.npz --prompt 'hello '
```

The tokenizer is character based, with pad, beginning, end, and unknown tokens.
It is deterministic and serializable. The examples train small models; they
are workflow checks, not demonstrations of general-purpose language ability.

## Training and checkpoints

### Batches and schedules

```python
from tensorforge.data import TensorDataset, DataLoader
from tensorforge.training import LRScheduler, clip_grad_norm_

loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=True, seed=7)
scheduler = LRScheduler(
    optimizer,
    schedule="cosine",
    warmup_steps=5,
    total_steps=100,
)

for features, targets in loader:
    optimizer.zero_grad()
    loss = criterion(model(features), targets)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
```

This fragment uses the regression `x`, `y`, `model`, `optimizer`, and
`criterion` from the neural-network example. Constructing a scheduler sets
the rate for the first update. Call `scheduler.step()` after each optimizer
update. Supported schedules are `constant`, `step`, and `cosine`, with optional
warmup.

To accumulate gradients, clear them once, backpropagate several appropriately
scaled losses, then call `optimizer.step()` once. For equal-size micro-batches,
divide each mean loss by their count. Unequal batches need sample-count
weighting. Clear gradients before the next update group.

`DataLoader` accepts a custom `collate_fn`. `pad_sequence` creates batch-first
padded arrays, and `padding_mask` marks the padding positions. Loader iteration
resumes an unfinished epoch; starting iteration after exhaustion creates a new
epoch. Its random generator is separate from `tf.manual_seed()`.

### Save and resume

```python
from tensorforge.training import save_checkpoint, load_checkpoint

save_checkpoint(
    "run.npz",
    model,
    optimizer,
    scheduler=scheduler,
    loader=loader,
    step=10,
    extra={"description": "my run"},
)

# Recreate the same architecture, optimizer parameter order, schedule, and
# dataset before restoring into a new process.
progress = load_checkpoint(
    "run.npz",
    model,
    optimizer,
    scheduler=scheduler,
    loader=loader,
)
print(progress["step"])
```

Save at an optimizer-update boundary after using all accumulated gradients.
Checkpoints contain model parameters, buffers, module modes, TensorForge random
state, and the supplied optimizer, scheduler, and loader states. Use `extra`
for model configuration, tokenizer state, and other plain metadata.

Dataset contents, arbitrary Python/NumPy random generators, custom collator
state, gradients, and attention caches are not automatically stored. Exact
continuation requires the same data and compatible environment; the tests
verify the next update with dropout and shuffled batches on the tested platform.

For weights only:

```python
tf.save(model.state_dict(), "weights.npz")
model.load_state_dict(tf.load("weights.npz"))
```

Storage uses JSON metadata and numeric NPZ arrays with pickle disabled.
Writes replace the destination atomically. Loading model state checks all keys,
shapes, and numeric types before applying it.

## Examples

Run examples from the repository after installing the package:

| File | Demonstrates |
| --- | --- |
| [01_scalar_autograd.py](examples/01_scalar_autograd.py) | A single-number computation graph |
| [02_tensor_basics.py](examples/02_tensor_basics.py) | Shapes, broadcasting, matrix multiplication, derivatives |
| [03_linear_regression.py](examples/03_linear_regression.py) | Mini-batch regression training |
| [04_xor_classifier.py](examples/04_xor_classifier.py) | A nonlinear dense classifier |
| [05_image_classifier.py](examples/05_image_classifier.py) | Convolution and pooling |
| [06_recurrent.py](examples/06_recurrent.py) | RNN, LSTM, and GRU training |
| [07_encoder_classifier.py](examples/07_encoder_classifier.py) | Transformer classification with padded inputs |
| [08_sequence_copy.py](examples/08_sequence_copy.py) | Encoder–decoder training and autoregressive copying |
| [09_language_model.py](examples/09_language_model.py) | GPT/modern decoder training, saving, and generation |
| [10_checkpoint.py](examples/10_checkpoint.py) | Restoring model and optimizer state |

Every default example includes checks for its expected behavior. Training
accuracy on tiny generated datasets validates the implementation, not
performance on unseen data.

## API reference

### `tensorforge`

- Creation: `Tensor`, `tensor`, `zeros`, `ones`, `randn`, `arange`.
- Random state: `manual_seed`, `get_rng_state`, `set_rng_state`.
- Combining/selecting: `cat(tensors, axis=0)`, `stack(tensors, axis=0)`, `where(condition, x, y)`.
- Graph control: `no_grad()`.
- Storage: `save(state, path)`, `load(path)`.
- Independent scalar engine: `Value`.

### Tensor methods

| Area | Methods and properties |
| --- | --- |
| Metadata | `shape`, `ndim`, `size`, `dtype`, `len(x)` |
| Arithmetic | `+`, `-`, `*`, `/`, fixed numeric `**`, `@`, `matmul`, unary `-`, `abs`, `sqrt` |
| Shape | `reshape`, `transpose(dim0, dim1)`, `permute`, `T`, `flatten`, `squeeze`, `unsqueeze`, `expand` |
| Selection | Slices/advanced indexing, `masked_fill`, `argmax` |
| Reduction | `sum`, `mean`, `amax`, `amin` with `axis` and `keepdims` |
| Functions | `exp`, `log`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `gelu`, `silu`, `softmax`, `log_softmax` |
| Autograd | `backward(gradient=None)`, `grad`, `zero_grad`, `detach`, `clone` |
| Conversion | `numpy()` gives a writable copy; `data` gives a read-only copy; `item()` extracts one entry |

### `tensorforge.nn`

- Structure: `Module`, `Parameter`, `Sequential`, `ModuleList`, `ModuleDict`.
- Basic layers: `Linear`, `Identity`, `Embedding`, `Flatten`, `Dropout`.
- Activations: `ReLU`, `LeakyReLU`, `Tanh`, `Sigmoid`, `GELU`, `SiLU`, `Softmax`.
- Normalization: `LayerNorm`, `RMSNorm`, `BatchNorm1d`, `BatchNorm2d`.
- Images: `Conv2d`, `MaxPool2d`, `AvgPool2d`.
- Recurrent layers: `RNN`, `LSTM`, `GRU`.
- Attention: `MultiheadAttention`, `FeedForward`, `RotaryEmbedding`.
- Position layers: `SinusoidalPositionEncoding`, `LearnedPositionEncoding`.
- Transformer layers: `TransformerEncoderLayer`, `TransformerDecoderLayer`.
- Transformer stacks: `TransformerEncoder`, `TransformerDecoder`, `Transformer`.
- Losses: `MSELoss`, `L1Loss`, `CrossEntropyLoss`, `BCEWithLogitsLoss`.
- `functional`: losses, common activations, and `scaled_dot_product_attention`.

Modules expose `parameters`, `named_parameters`, `buffers`, `named_buffers`,
`register_buffer`, `zero_grad`, `train`, `eval`, `state_dict`, and
`load_state_dict`.

Loss reductions are `"mean"`, `"sum"`, or `"none"`. MSE/L1/binary losses require
matching shapes. Cross-entropy accepts `(..., classes)` raw logits and integer
targets shaped `(...)`, with `ignore_index=-100` and optional label smoothing.
Its mean excludes ignored positions; an entirely ignored batch returns zero.
Do not apply softmax before cross-entropy.

```python
import numpy as np

labels = tf.tensor([0, 1, 0], dtype=np.int64)
criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
```

Recurrent inputs have shape `(batch, time, input_size)`. RNN and GRU return
`(sequence, hidden)`; LSTM returns `(sequence, (hidden, cell))`. Hidden and cell
states have shape `(batch, hidden_size)`. Layers are single-layer and
unidirectional; compose separate modules for deeper networks.

Image layers expect `(batch, channels, height, width)`. Convolution and pooling
use square integer kernels and integer strides. Conv2d has integer zero
padding; pooling has no padding. LayerNorm and RMSNorm normalize the final
feature axis. BatchNorm1d accepts `(N,C)`/`(N,C,L)`; BatchNorm2d accepts NCHW.

### Other modules

| Module | Public components |
| --- | --- |
| `optim` | `SGD`, `Adam`, `AdamW`; each has `step`, `zero_grad`, `state_dict`, `load_state_dict` |
| `training` | `clip_grad_norm_`, `LRScheduler`, `save_checkpoint`, `load_checkpoint` |
| `data` | `Dataset`, `TensorDataset`, `DataLoader`, `pad_sequence`, `padding_mask` |
| `text` | `CharacterTokenizer`, `TextDataset` |
| `models` | `LanguageModelConfig`, `CausalLanguageModel` |

SGD supports momentum. SGD and Adam use L2 weight penalties; AdamW uses
decoupled weight decay. Optimizers accept a flat collection of trainable leaf
tensors, not parameter groups.

## Testing and packaging

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m pip check
python -m build
```

The tests include numerical gradient checks, shared graphs, shape/error cases,
optimizer equations, masking, causal isolation, recurrent continuation,
cached/uncached agreement, checkpoint restoration, and runnable training
examples. To see output from successful examples:

```bash
python -m pytest tests/test_examples.py -rP
```

Build outputs:

```text
dist/tensorforge-0.1.0-py3-none-any.whl
dist/tensorforge-0.1.0.tar.gz
```

The wheel contains the library. The source archive also includes examples and
tests. Install the wheel in another environment with:

```bash
python -m pip install /absolute/path/to/dist/tensorforge-0.1.0-py3-none-any.whl
```

GitHub Actions checks Python 3.10, 3.11, and 3.12. Local verification results
and remote CI status are distinct; the badge links to the actual workflow.

## Project layout

```text
tensorforge/
  tensor.py          Tensor operations and automatic differentiation
  scalar.py          Single-number automatic differentiation
  nn/
    modules.py       Basic layers, normalization, and model structure
    functional.py    Losses and attention operations
    vision.py        Convolution and pooling
    recurrent.py     RNN, LSTM, and GRU
    transformer.py   Position layers, attention, and Transformer blocks
  models.py          Causal language models and generation
  optim.py           Parameter update rules
  training.py        Schedules, clipping, and training checkpoints
  data.py            Datasets, batching, and padding
  text.py            Character tokenizer and text windows
  serialization.py   Portable state files
examples/            Numbered runnable examples
tests/               Automated checks
.github/workflows/   Continuous integration
```

## Scope and limitations

- CPU/NumPy backend only. No CUDA, distributed training, automatic mixed
  precision, graph compilation, sparse tensors, or optimized custom kernels.
- Designed for small models. No throughput, production-readiness, or
  billion-parameter training claim.
- Numeric inputs default to **float64**, including integer Python lists.
  Explicit integer/bool tensors are available without gradients. Float32 can
  be requested, but mixed operations may promote to float64.
- Gradients are NumPy arrays. Tensor gradients accumulate across calls;
  `Value.backward()` resets its scalar graph each call. Clear parameter
  gradients between independent optimizer updates.
- First-order differentiation only. No higher-order graph, hooks, general
  in-place operations, or PyTorch autograd extension compatibility.
- `backward()` without an upstream gradient accepts one-element outputs.
  Other outputs require a gradient of exactly the same shape.
- Outputs retain their graph until released. Store `loss.item()` in logs,
  rather than retaining every loss tensor. Updates invalidate existing graphs;
  recompute the forward pass after an optimizer step.
- Public array access copies data. There is no storage/view aliasing API.
  Reductions use `axis`/`keepdims`; `.T` reverses all axes. These conventions
  differ from parts of PyTorch's API.
- Real numbers only. Fixed-power exponents are finite Python numbers. Log
  requires positive inputs. ReLU/absolute value use zero gradient at zero;
  maximum reductions share ties, while max pooling chooses the first tie.
- No convolution groups/dilation, transposed convolution, recurrent packed
  sequences, multi-process loading, or pretrained model import.
- Attention uses a dense score matrix, so memory grows quadratically with
  sequence length. Cache use is inference-only. No FlashAttention or sliding
  window cache eviction.
- Tokenization is character based, not BPE. Model/tokenizer architecture
  metadata must be saved explicitly when using generic checkpoint functions.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New differentiable operations should
include independent value checks, numerical gradient checks, and shape/error
tests. Keep source names descriptive and examples small and reproducible.

## License

[MIT](LICENSE).
