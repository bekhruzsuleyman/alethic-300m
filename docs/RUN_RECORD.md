# Alethic-300M run record

## M0: repo skeleton, model, loader, all tests (laptop CPU), 2026-11-09

**Gate:** every test in the plan's table is green. **Result: passed.** 30 passed, 0 failed. Cost $0.

### Machine and software

| Item | Value |
| --- | --- |
| Laptop | Windows 11, 8 logical CPUs, 3.65 GB RAM (about 0.5 GB free while testing) |
| Python | 3.12.10 (installed with `py install 3.12`; the laptop only had 3.14 / 3.15) |
| torch | 2.15.0+cpu |
| sentencepiece | 0.2.3 |
| numpy / pyarrow / regex / huggingface_hub / PyYAML / pytest | 2.5.4 / 26.0.1 / 2026.11.9 / 2.4.0 / 6.0.3 / 9.2.0 |

### Commands

```text
py install 3.12
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install numpy pyarrow sentencepiece regex huggingface_hub pyyaml pytest
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\python -m pytest -v
```

### Config hashes (SHA-256)

| File | SHA-256 |
| --- | --- |
| configs/tokenizer.yaml | a18af72def5ad1ace0a47e50b2629e045c5f7bb0cb363da4af8e103e1521d6ae |
| configs/data.yaml | f45bb084fc4c58e74b7f218d26f7f5477aa0e0fec72a46a1bf71fc2b551f9c17 |
| configs/model_sgca.yaml | 5c16a1bef1edb34b9a4ce59093e08501a66cac182e4aa4169e2b1bba601f6463 |
| configs/model_gqa.yaml | ffb0ae450703e47032876a3af9954c61306194dfa597f20955074669ec01ee20 |
| configs/train.yaml | 4f24f086832b00462f5ad325904b1e3c0bffe534fe2d2b990ab931525eebf114 |

There is no tokenizer hash yet: the tokenizer is trained and frozen in M1.

### Test results (full suite: 49.5 s of tests, 54 s wall; the limit is 5 min)

| Plan test | Result | Notes |
| --- | --- | --- |
| test_tokenizer_roundtrip | pass | 20 hard cases plus 2,000 random strings (all code points except surrogates, plus U+2581 and U+E000), 100% |
| test_u2581_roundtrip (extra) | pass | literal U+2581 and the escape marker U+E000: 11 cases plus 2,000 random strings, 100% |
| test_newline_bytes | pass | `"\n"` → `[<0x0A>]` (id 78) |
| test_indentation | pass | newline + 4 / 8 / 12 spaces → `<0x0A>` + one learned piece = 2 tokens |
| test_vocab_ids | pass | ids 0-3 control, 4-67 `<sp00>`..`<sp63>`, 68-323 bytes, learned from 324 |
| test_specials | pass | plus an extra test: refusing a save after the file is renamed under a running registry |
| test_shard_atomic | pass | crash injected halfway through a shard write, plus 3 outside kills mid-build; every rerun gives byte-identical shards |
| test_split_function | pass | 0.1% ± 0.05% of 200,000 keys; same repo gives the same side in both code sources |
| test_uzbek_apostrophe | pass | all 6 variants, Latin and Cyrillic, counts checked |
| test_param_counts | pass | both models 302,908,416 (built on the meta device, so no RAM used) |
| test_causality | pass | exact equality before t, for both models |
| test_sgca_gate | pass | block output equals hand-written RMSNorm, RoPE, GQA, X + C and FFN within 1e-5 |
| test_loader_permutation | pass | 40 windows over 3 uneven shards; each window once; resume equal |
| test_resume_exact | pass | 20 straight vs 10 + resume + 10: max difference ≤ 1e-6 (both models) |
| test_overfit | pass | loss < 0.1 within 200 steps (both models) |
| test_cost_guard | pass | via the real CLI; stops at step 1, writes a checkpoint, `--resume auto` continues |

Extra tests: config placeholder refusal, 91,553 steps, LR schedule end points, BPB formula,
torn `state.jsonl` line, shard cutting and sidecars, init scales, wrong tokenizer hash refused.

### Worked numbers checked by tests

- Parameters: GQA layer = 2,621,440 + 7,864,320 + 2,048 = 10,487,808; × 24 = 251,707,392;
  + 51,200,000 + 1,024 = **302,908,416**. SGCA adds 1,050,624 per layer and its FFN loses 1,050,624.
- FLOPs per token: 6 × 302,908,416 + 12 × 24 × 1,024 × 4,096 = 1,817,450,496 + 1,207,959,552 = **3,025,410,048**.
- Steps: ceil(6,000,000,000 / 65,536) = **91,553**.

### Measured speed

None that matters for later stages: this laptop has no GPU and too little free RAM (0.46 GB) to run
the full 302M model even forward, so speed and MFU are first measured in M4. The full-size models are
checked by building them on the meta device (exact shapes, no memory).

### Surprises

1. **Literal U+2581 (`▁`) did not round-trip; fixed with an escape.** SentencePiece turns spaces into
   `▁` internally, so a real `▁` decoded as a space. `escape_whitespaces=false` would avoid it, but
   sentencepiece 0.2.3's Unigram trainer stops with `unigram_model_trainer.cc(828)
   [normalizer_spec_.escape_whitespaces()]`. Fix (owner asked for 100% round trip): normalization is
   identity plus two escape rules stored inside the .model file. A real U+2581 becomes U+E000 "0" and a
   real U+E000 becomes U+E000 "1"; decoding reverses both, like an escaped backslash `\\` in strings. No other character is
   changed (no NFKC). Cost: a real `▁` takes 4 tokens (3 bytes of U+E000 plus "0"); it is rare.
   Caveat for later: llama.cpp ignores SentencePiece normalization rules, so an app on llama.cpp must
   apply the same two-rule escape itself (training data and the Python codec are exact).
2. **`<spNN>` text can be typed.** The slots are `user_defined_symbols`, as the plan says, so the
   literal text `<sp05>` in user input encodes to the special id. Names such as `<thinking>` are
   safe. `control_symbols` would block this, but then `<sp00>` text would no longer encode to one id,
   which contradicts `test_vocab_ids`. **Owner decision (2026-11-09): keep `user_defined_symbols` as
   the plan says.** The app that serves the model should treat user text containing `<sp` with care
   once a slot is in use (first planned use: M8 image tokens).
3. **Whitespace symbols must be written with `▁`.** `user_defined_symbols` that contain real spaces are
   accepted but never match. Written as `▁▁▁▁…` they work (checked with 16 and 20). `train.py` now
   refuses the space form. The tiny tokenizer learned 4, 8 and 12-space pieces without this fallback.
4. pytest's default temp folder was blocked in my sandbox, so tests ran with `--basetemp`. A normal
   shell does not need this.

### Not in M0 (comes later)

Tokenizer sample builder and evaluation (M1); fetch, dedup, decontamination, parallel encoder and
data manifest (M2); validation BPB driver and periodic eval in the training loop (before M4).
Those modules exist as stubs that raise `NotImplementedError` and name their milestone.

### Rented servers

None were started. Nothing is running or billing.
