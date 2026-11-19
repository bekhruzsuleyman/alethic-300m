import torch
import torch.nn as nn

from alethic.sft import train as sft


def test_load_base_accepts_the_best_checkpoint_of_an_earlier_sft_run(tmp_path):
    """Stage B starts from stage A's output folder (best/model.pt), which has no manifest or sha."""
    a, b = nn.Linear(3, 3), nn.Linear(3, 3)
    (tmp_path / "best").mkdir()
    torch.save({"model": a.state_dict(), "epoch": 1, "val_loss": 1.0}, tmp_path / "best" / "model.pt")
    info = sft.load_base(b, tmp_path)
    assert torch.equal(a.weight, b.weight) and info["n_tensors"] == 2 and info["step"] is None
