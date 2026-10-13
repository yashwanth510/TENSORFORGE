# Contributing

Keep TensorForge small, readable, and explicit about its supported behavior.
Use descriptive source filenames and numbered runnable examples. Comments
should explain non-obvious decisions and derivative formulas.

## Development

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m build
git diff --check
```

For comparisons against native PyTorch layers and the GPT-style reference,
install CPU PyTorch and the optional reference extra:

```bash
python -m pip install 'torch>=2.5,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,reference]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/test_pytorch_reference.py -v
```

Match float64 inputs, weights, loss reductions, and layer settings. Keep
dropout disabled and compare input/state and parameter gradients as well as
outputs. PyTorch remains optional outside the dedicated reference CI job;
retain independent finite-difference checks for derivatives.

Before submitting a change:

- Test expected values independently of the implementation.
- Check derivatives with finite differences, including broadcasting and shared inputs.
- Test meaningful shape and input errors.
- Update the README when public behavior changes.
- Keep examples deterministic and independent of downloaded datasets.

Do not commit environments, credentials, local editor settings, checkpoints,
or build outputs. Do not weaken assertions to hide unexplained failures.
