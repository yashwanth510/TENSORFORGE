# TensorForge

An educational deep-learning library built from scratch, inspired by PyTorch.

We implement scalar automatic differentiation and the framework ourselves.
For the initial tensor implementation, NumPy provides storage and numerical
operations. TensorForge is not a replacement for PyTorch.

## First milestone

- [x] Create a local Python virtual environment.
- [x] Add a README and Git ignore rules.
- [ ] Implement scalar arithmetic and automatic differentiation.
- [ ] Implement tensor creation, shapes, and dtypes.
- [ ] Implement elementwise arithmetic and broadcasting.
- [ ] Implement indexing, reshape, and transpose.
- [ ] Implement reductions and matrix multiplication.
- [ ] Test the implemented behavior.
- [ ] Document examples and push the project to GitHub.

Tensor automatic differentiation comes after this milestone.

## Learning workflow

Build in small steps and create files only when needed. For each feature:

1. Explain the problem and derive the relevant mathematics.
2. Work through a numerical example.
3. Describe the algorithm in plain language.
4. Explain every new line of code, including why it exists.
5. Trace values through execution.
6. Verify results and meaningful edge cases.

Save detailed explanations in lesson files as implementation progresses.
Include exact Bash commands for learner-run steps. Record dependencies when
they are introduced and update this README as features become available.

## Development setup

Initial development uses Python 3.11.

Create an environment once:

```bash
python3 -m venv .venv
```

Activate it in each new terminal:

```bash
source .venv/bin/activate
```

No third-party dependencies are required at the current setup stage.
We will introduce pytest for testing and NumPy for tensor operations.

## Current status

The project environment and documentation are set up. Library implementation
has not started yet.
