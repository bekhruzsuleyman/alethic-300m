"""Config loading, placeholder refusal, step count and learning-rate schedule."""
from __future__ import annotations

import math

import pytest

from alethic.config import ConfigError, load_config
from alethic.eval.bpb import bits_per_byte
from alethic.train.loop import total_steps
from alethic.train.optim import lr_at
from conftest import CONFIGS


def test_all_configs_load():
    for name in ("tokenizer.yaml", "data.yaml", "model_sgca.yaml", "model_gqa.yaml", "train.yaml"):
        assert load_config(CONFIGS / name)


def test_placeholders_refused(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("a: 1\nsources:\n  python: {revision: REPLACE_WITH_PINNED_COMMIT}\nk: [ok, YOUR_TOKEN]\n")
    with pytest.raises(ConfigError) as err:
        load_config(p)
    assert "sources.python.revision" in str(err.value) and "k[1]" in str(err.value)


def test_steps_and_schedule():
    cfg = load_config(CONFIGS / "train.yaml")
    assert total_steps(cfg) == 91_553 == math.ceil(6_000_000_000 / 65_536)
    o = cfg["optim"]
    peak, low, warm = o["peak_lr"], o["min_lr"], o["warmup_steps"]
    assert (peak, low, warm) == (4e-4, 4e-5, 1000)
    assert lr_at(0, peak, low, warm, 91_553) == pytest.approx(peak / 1000)
    assert lr_at(999, peak, low, warm, 91_553) == pytest.approx(peak)
    assert lr_at(91_553, peak, low, warm, 91_553) == pytest.approx(low)
    mid = (1000 + 91_553) // 2
    assert lr_at(mid, peak, low, warm, 91_553) == pytest.approx((peak + low) / 2, rel=1e-3)


def test_bits_per_byte():
    assert bits_per_byte(math.log(2) * 100, 100) == pytest.approx(1.0)
