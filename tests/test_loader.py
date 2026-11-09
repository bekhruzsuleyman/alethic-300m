"""Loader: every window exactly once per epoch, windows cross shards correctly, resume is exact."""
from __future__ import annotations

import numpy as np
import pytest

from alethic.data.loader import LoaderError, WindowLoader
from alethic.data.shards import to_tokens


def _shards(tmp_path, sizes):
    """Shards whose tokens are 0, 1, 2, ... across the whole stream, so windows are checkable."""
    paths, start = [], 0
    for i, n in enumerate(sizes):
        p = tmp_path / f"s{i}.bin"
        p.write_bytes(to_tokens(np.arange(start, start + n) % 65536).tobytes())
        paths.append(p)
        start += n
    return paths


def test_loader_permutation(tmp_path):
    """Every window appears exactly once; resuming at step k gives the same batches."""
    seq, batch = 8, 4
    paths = _shards(tmp_path, [100, 7, 214])            # 321 tokens -> (321 - 1) // 8 = 40 windows
    loader = WindowLoader(paths, seq, batch, seed=42)
    assert loader.n_windows == 40 and loader.steps_per_epoch == 10

    seen = np.concatenate([loader.window_ids(s) for s in range(loader.steps_per_epoch)])
    assert sorted(seen.tolist()) == list(range(40))    # each window exactly once
    assert seen.tolist() != list(range(40))            # and actually shuffled

    for s in range(loader.steps_per_epoch):              # contents: window w = tokens w*8 .. w*8+8
        b = loader.batch(s)
        assert b.shape == (batch, seq + 1) and b.dtype == np.int64
        for row, w in zip(b, loader.window_ids(s)):
            assert row.tolist() == list(range(w * seq, w * seq + seq + 1))

    resumed = WindowLoader(paths, seq, batch, seed=42)  # a fresh process resuming at step k
    for k in (3, 9, 0, 10, 25):
        assert np.array_equal(resumed.batch(k), loader.batch(k))

    second = np.concatenate([loader.window_ids(s) for s in range(10, 20)])
    assert sorted(second.tolist()) == list(range(40))   # past one pass: a new full permutation
    other = WindowLoader(paths, seq, batch, seed=43)
    assert not np.array_equal(other.window_ids(0), loader.window_ids(0))


def test_loader_too_small(tmp_path):
    with pytest.raises(LoaderError):
        WindowLoader(_shards(tmp_path, [20]), 8, 4, seed=0)
