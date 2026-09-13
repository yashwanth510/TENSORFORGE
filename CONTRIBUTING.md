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

Before submitting a change:

- Test expected values independently of the implementation.
- Check derivatives with finite differences, including broadcasting and shared inputs.
- Test meaningful shape and input errors.
- Update the README when public behavior changes.
- Keep examples deterministic and independent of downloaded datasets.

Do not commit environments, credentials, local editor settings, checkpoints,
or build outputs. Do not weaken assertions to hide unexplained failures.
