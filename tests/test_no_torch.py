"""The data and tokenizer commands must work without torch (the Modal CPU image does not install it)."""
from __future__ import annotations

import subprocess
import sys

SCRIPT = r"""
import sys
class BlockTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is blocked in this test")
        return None
sys.meta_path.insert(0, BlockTorch())
from alethic import cli
import alethic.tokenizer.build, alethic.data.fetch, alethic.tokenizer.sample, alethic.tokenizer.evaluate
assert cli.main(["data-build"]) == 2                  # goes through main() and its error handling
assert "torch" not in sys.modules
print("ok")
"""


def test_cli_without_torch():
    out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-2000:]
