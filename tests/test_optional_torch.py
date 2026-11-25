from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_numpy_backend_works_when_torch_import_is_blocked() -> None:
    root = Path(__file__).resolve().parents[1]
    script = r'''
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch intentionally blocked")
        return None

sys.meta_path.insert(0, BlockTorch())

import numpy as np
import spheredd

assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
points, weights, info = spheredd.spherical_design(np.eye(4), 1, design_order=6)
assert points.shape == (42, 4)
assert info["backend"] == "numpy"
assert spheredd.torch_available() is False
print("ok")
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.stdout.strip() == "ok"
