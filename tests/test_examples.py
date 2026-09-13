"""Run documented entry points as subprocesses to test actual user workflows."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path", sorted((ROOT / "examples").glob("[0-9][0-9]_*.py")), ids=lambda p: p.name
)
def test_example(path):
    completed = subprocess.run(
        [sys.executable, str(path)], cwd=ROOT, capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    print(completed.stdout)


def test_modern_language_model(tmp_path):
    checkpoint = tmp_path / "modern.npz"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/09_language_model.py"),
            "--variant",
            "modern",
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    print(completed.stdout)
    restored = subprocess.run(
        [sys.executable, str(ROOT / "examples/09_language_model.py"), "--load", str(checkpoint)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert restored.stdout.strip() == completed.stdout.strip().splitlines()[-1]
