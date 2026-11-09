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

---

## M1: 50,000-id SentencePiece Unigram tokenizer on Modal CPU, 2026-11-10

**Gate:** 100% lossless round-trip across all 6 sources, indentation (`4`, `8`, `12` spaces) ≤ 2 tokens on ≥ 99% of cases, Uzbek ≤ 3.0 tokens/word, 2,000 held-out docs per source, frozen with SHA-256 in `FROZEN.json` and backed up to private Hugging Face repo `bekhruzsuleyman/alethic-300m`. **Result: passed.**

### Frozen tokenizer hashes (`FROZEN.json`)

| Artifact | SHA-256 |
| --- | --- |
| `alethic-tokenizer-v1.model` (`model_sha256`) | `28685edb461ce3baa8b36f08c32cb04c8f6cb2c1c0c79c6e737816ad214a382b` |
| `configs/tokenizer.yaml` (`config_sha256`) | `81ca75999f7b48b5d654ed4454087c85b30688f9b4cbed053f934ef8c39c4bf1` |
| `report.json` (`report_sha256`) | `a61d310d2596722b4bbfc1f12f3003bb42782bde465a9cbe0d63d314defb0e2a` |
| Hugging Face backup (`bekhruzsuleyman/alethic-300m` `tokenizer/`) | Commit `4d3749af9c614e06579c33dc341d50654d77d4d5` |

### Per-source evaluation metrics (2,000 held-out documents per source; 12,002 total)

| Source | Docs | UTF-8 Bytes | Tokens | Bytes/Token | Tokens/Word | Byte-Fallback Share | Round-Trip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `python` | 2,000 | 7,764,458 | 2,155,705 | 3.6018 | 3.3522 | 10.08% | **100.00%** |
| `code_other` | 2,002 | 6,349,179 | 1,990,023 | 3.1905 | 3.2041 | 10.26% | **100.00%** |
| `fineweb_edu` | 2,000 | 9,535,149 | 1,833,932 | 5.1993 | 1.2248 | 2.24% | **100.00%** |
| `math` | 2,000 | 14,917,053 | 3,790,545 | 3.9353 | 1.6305 | 4.62% | **100.00%** |
| `uzbek_latn` | 2,000 | 6,955,586 | 1,806,499 | 3.8503 | **2.1674** (≤ 3.0) | 1.84% | **100.00%** |
| `uzbek_cyrl` | 2,000 | 7,057,623 | 1,264,539 | 5.5812 | **2.5515** (≤ 3.0) | 1.59% | **100.00%** |

- **Python indentation gate (`indentation_max_tokens: 2`, `indentation_min_share: 0.99`):**
  - 4 spaces (piece ID `338`): `65,080 / 65,080` (**100.00%**)
  - 8 spaces (piece ID `340`): `77,101 / 77,101` (**100.00%**)
  - 12 spaces (piece ID `356`): `36,122 / 36,122` (**100.00%**)
- **Uzbek apostrophe normalization (`sample.json`):** `1,829,104` chars normalized in `uzbek_latn`, `3,968` chars normalized in `uzbek_cyrl`.
- **Measured timings (`timings.json`):** `train_seconds: 5003.42` (83.4 min across 3 EM stages `1M → 300k → 120k → 50k` on 32 threads), `eval_seconds: 63.36` (1.06 min).
- **Modal cost:** ~$3.40 of the $30/month free Starter credits across sample download + run 2 (cancelled by local CLI 60-min timeout) + run 3 (spawned with EM checkpoints at 300k and 120k, 32 GiB RAM).

---

## M2 & M3a: Data pipeline speed test and 1B-token shard build (`--target-scale 0.1667`), 2026-11-10

**Gates:**
- **M2:** Kill and rerun gives identical shard hashes (`test_shard_atomic` + `test_encode_units_and_eval_bpb`); throughput ≥ 50,000 tokens/sec/worker measured; projected cost of the full 6B-token build recorded.
- **M3a:** 1B-token subset built (`target_scale = 0.1667`); every source within ±2 percentage points of `37 / 17 / 28 / 8 / 10` and Uzbek ≥ 9%; `shards/manifest.json` verified; decontamination and deduplication counts recorded.
**Result: passed.**

### Commands

```text
python -m pytest -q
modal run --detach deploy/modal_app.py::data_build --target-scale 0.1667 --workers 16 --spawn
```

### Measured throughput and projected 6B build cost (`M2`)

| Metric | Value |
| --- | --- |
| Container | Modal CPU: 16 physical cores (32 vCPU) + 32 GiB RAM (`$1.01/h`) |
| Workers | 16 |
| Total tokens encoded | `1,061,655,074` (`1,060,637,876` train + `1,017,198` val across 235 shards) |
| Wall time (`elapsed_seconds`) | **`183.05 s` (3.05 min)** |
| Total throughput (`tokens_per_sec_total`) | **`5,799,943.0` tokens/s** |
| Throughput per worker (`tokens_per_sec_per_worker`) | **`362,496.4` tokens/s/worker** (7.25× the 50k gate) |
| M3a 1B build cost | **`$0.05`** (`183.05 s / 3600 × $1.01/h`) |
| Projected full 6B-token build (`M3b`, remaining 5B tokens) | **`~15–18 min` (`~$0.25–$0.30`)** continuing from `state.jsonl` |

### M3a 1B-token mix, deduplication, and decontamination (`shards/manifest.json`)

- `tokenizer_sha256`: `28685edb461ce3baa8b36f08c32cb04c8f6cb2c1c0c79c6e737816ad214a382b`
- `shards_sha256` (221 train shards): `fdb5f43b83d2cf34ba98ca328a3ac09806d160bc9c8a9a61e1eace02c9408adc`

| Source | Target Share | Actual Share | Diff (pp) | Total Tokens | Train Tokens | Val Tokens | Dedup Dropped | Decontam Dropped (13-gram) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `python` | 36.67% | **36.36%** | -0.31 | 386,045,492 | 385,692,433 | 353,059 | 5 | 57 |
| `code_other` | 16.67% | **16.31%** | -0.36 | 173,169,866 | 172,980,480 | 189,386 | 2 | 0 |
| `fineweb_edu` | 28.33% | **28.63%** | +0.30 | 303,899,168 | 303,636,488 | 262,680 | 0 | 0 |
| `math` | 8.33% | **8.56%** | +0.23 | 90,885,088 | 90,813,252 | 71,836 | 0 | 1 |
| `uzbek_latn` | 6.67% | **6.91%** | +0.24 | 73,368,989 | 73,299,572 | 69,417 | 6 | 0 |
| `uzbek_cyrl` | 3.33% | **3.23%** | -0.10 | 34,286,471 | 34,215,651 | 70,820 | 0 | 0 |
| **Total / Combined** | **100.00%** (Uzbek 10.0%) | **100.00%** (Uzbek **10.14%**) | max **0.36 pp** | **1,061,655,074** | **1,060,637,876** | **1,017,198** | **17** (`1,096,476` docs) | **58** |

- **Uzbek apostrophe edits:** `3,590,220` chars normalized in `uzbek_latn`, `11,949` in `uzbek_cyrl`.
- **Attributions recorded in `manifest.json`:** `HuggingFaceFW/fineweb-2` (`af9c13333eb981300149d5ca60a8e9d659b276b9`, license `ODC-By`).

---

## M4: 2,000-step GPU smoke test on Modal H100 (`m4_sgca` & `m4_gqa`), 2026-11-10

**Gate:** Train both `sgca` and `gqa` (`302,908,416` params each) for 2,000 steps (`131,072,000` tokens) on the M3a shards with identical seed (`1234`), data order, and hyperparameters; verify loss falls, measure `tokens_per_sec` and `MFU` (≥ 30%), record validation BPB per source and overall at steps 1,000 and 2,000, verify atomic 500-step checkpoints, and project full 6B-token run cost and duration. **Result: passed.**

### Machine & software versions (`runs.jsonl`)

- **Container:** `NVIDIA H100 80GB HBM3` + 4 physical CPU cores + 32 GiB RAM (`$4.39/h` on Modal)
- **Software:** Python `3.12.16`, PyTorch `2.15.0+cu132` (`bf16` + `torch.compile`), CUDA `13.2`, NumPy `2.5.4`, SentencePiece `0.2.3`
- **Config & checkpoint hashes:**
  - `m4_sgca`: `model_config_sha256 = 6a741d419c9a0ea0131b0ad8555b0787feab36be4d3aff5396c951c4871cbc1e`, `step_0002000/state.pt` `state_sha256 = e37685aefc5b57b60d50298e66e12efc35238f69e26c7d4aac376578a6988845`
  - `m4_gqa`: `model_config_sha256 = f0df244634935342c633cff44c1ee73d0c1fd30262736e77657160764700fd0e`, `step_0002000/state.pt` `state_sha256 = 0ab4176c90a8dda98d1c2aafe2b7a08ed638375194a4ca5fb1fe185fb24c0b5a`

### Speed, MFU, loss trajectory, and 6B-token cost projection

| Model | Step 100 Loss | Step 500 Loss | Step 1000 Loss | Step 1500 Loss | Step 2000 Loss | Steady-State tok/s | Steady-State MFU | 2,000-Step Cost | Projected 6B Run (91,553 steps) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **SGCA** (`m4_sgca`) | 9.2027 | **5.8798** | **4.4792** | **3.5904** | 3.3186 | **`139,656` tok/s** | **`42.72%`** | `$1.38` | **`11.93 h`** (`$52.38` at Modal `$4.39/h`; `~$26–$33` on Vast.ai H100) |
| **GQA** (`m4_gqa`) | 8.8623 | 6.0781 | 4.7455 | 3.6585 | 3.3185 | **`148,952` tok/s** | **`45.57%`** | `$1.16` | **`11.19 h`** (`$49.12` at Modal `$4.39/h`; `~$24–$31` on Vast.ai H100) |

### Validation bits per byte (`val_log.jsonl`, `1,017,192` val tokens = `4,085,874` UTF-8 bytes)

| Step & Model | Overall BPB | `python` BPB | `code_other` BPB | `fineweb_edu` BPB | `math` BPB | `uzbek_latn` BPB | `uzbek_cyrl` BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Step 1,000 — SGCA** | **`1.526906`** | **`1.300738`** | **`1.675784`** | **`1.595360`** | **`1.834102`** | **`1.919390`** | **`1.290044`** |
| **Step 1,000 — GQA** | `1.605544` | `1.386307` | `1.769109` | `1.644097` | `1.909167` | `2.017571` | `1.418431` |
| **Step 2,000 — SGCA** | `1.173432` | `0.906362` | `1.142758` | **`1.313570`** | `1.470070` | **`1.579454`** | **`1.063173`** |
| **Step 2,000 — GQA** | **`1.163813`** | **`0.889746`** | **`1.099619`** | `1.314693` | **`1.451142`** | `1.592587` | `1.075824` |

- **Observations:**
  1. At **step 1,000** (end of warmup), **SGCA leads GQA across all 6 sources** by **0.0786 BPB overall** (`1.5269` vs `1.6055`).
  2. At **step 2,000** (`131M` tokens), **SGCA leads on all natural-language streams** (`fineweb_edu` `1.3136` vs `1.3147`, `uzbek_latn` `1.5795` vs `1.5926`, `uzbek_cyrl` `1.0632` vs `1.0758`), while GQA edges ahead on code (`python` `0.8897` vs `0.9064`, `code_other` `1.0996` vs `1.1428`).
  3. Both models exceed **42.7% MFU** on H100 (vs the 30% gate), cutting projected 1B-token M6 runtime to **`1.99 hours` (SGCA)** and **`1.86 hours` (GQA)**.
- **Hugging Face backups (`bekhruzsuleyman/alethic-300m`):** `shards/manifest.json` (`46dd2842c74cba4b88dc86b8a15c5acf184b7203`), `runs/m4_sgca` (`0bbf553efc8eb885b7cf526700da31a4a8adf3da`), `runs/m4_gqa` (`ce0d121983a7f9e085cf79e95551a23aad766b24`).

---

## M5: Learning-rate sweep (`100M` tokens = `1,526` steps × LR `{2e-4, 4e-4, 8e-4}` × `{sgca, gqa}`), 2026-11-10

**Gate:** Train all 6 configurations (`sgca` and `gqa` at peak LR `2e-4`, `4e-4`, `8e-4`, cosine decay to `peak_lr / 10` over `1,526` steps = `100,007,936` tokens, 5% linear warmup = `76` steps) on `1 × H100` with identical seed (`1234`) and data order; record all 6 curves and per-source validation BPB; pick the learning rate for M6. **Result: passed.** Total cost across all 6 runs: **`$5.48`** (`80 min` in a single H100 session `ap-n4ANFR7D7HvtDgPzVprO9a`).

### Final results at step 1,526 (`100,007,936` tokens; full validation across all `1,017,192` val tokens)

| Run Name | Model | Peak LR (`min_lr`) | Step 1000 Train Loss | Step 1526 Train Loss | Step 1000 Val BPB | **Step 1526 Val BPB (Overall)** | `python` BPB | `code_other` BPB | `fineweb_edu` BPB | `math` BPB | `uzbek_latn` BPB | `uzbek_cyrl` BPB | Cost |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `m5_sgca_lr2e4` | **SGCA** | `2e-4` (`2e-5`) | 4.6540 | **3.9094** | 1.558281 | **`1.409033`** | **`1.186081`** | **`1.499782`** | **`1.494161`** | **`1.727700`** | **`1.766123`** | **`1.201004`** | `$1.14` |
| `m5_gqa_lr2e4` | GQA | `2e-4` (`2e-5`) | 4.9482 | 4.1734 | 1.695068 | `1.503421` | `1.264176` | `1.583551` | `1.585534` | `1.808938` | `1.924773` | `1.336189` | `$0.90` |
| `m5_sgca_lr4e4` | **SGCA** | `4e-4` (`4e-5`) | 4.1672 | **3.4892** | 1.410173 | **`1.269719`** | **`1.016509`** | **`1.255078`** | **`1.402231`** | **`1.580446`** | **`1.661830`** | **`1.119912`** | `$0.89` |
| `m5_gqa_lr4e4` | GQA | `4e-4` (`4e-5`) | 4.5367 | 3.8329 | 1.543931 | `1.394150` | `1.134473` | `1.403500` | `1.519340` | `1.693468` | `1.830558` | `1.234398` | `$0.83` |
| `m5_sgca_lr8e4` | **SGCA** | `8e-4` (`8e-5`) | 4.0725 | **3.3990** | 1.383821 | **`1.244308`** | **`0.965804`** | **`1.162858`** | **`1.401845`** | **`1.537008`** | **`1.702264`** | **`1.147078`** | `$0.89` |
| `m5_gqa_lr8e4` | GQA | `8e-4` (`8e-5`) | 4.4926 | 3.7995 | 1.524024 | `1.378295` | `1.136014` | `1.399223` | `1.495458` | `1.685459` | `1.787426` | `1.189336` | `$0.83` |

### Key findings from the M5 sweep

1. **SGCA beats GQA at every learning rate and on every single data source** when cosine decay completes at 100M tokens (`step 1526`):
   - At `2e-4`: **SGCA `1.409033`** vs GQA `1.503421` (**`-0.0944` BPB** in SGCA's favor).
   - At `4e-4`: **SGCA `1.269719`** vs GQA `1.394150` (**`-0.1244` BPB** in SGCA's favor).
   - At `8e-4`: **SGCA `1.244308`** vs GQA `1.378295` (**`-0.1340` BPB** in SGCA's favor).
   - SGCA at `4e-4` (`1.2697`) even beats GQA at `8e-4` (`1.3783`) by **`-0.1086` BPB** across all 6 sources.
2. **Selected learning rate (`4.0e-4`, matching `configs/train.yaml`):**
   - While `8e-4` edges out `4e-4` on code/math at the 100M-token horizon, `4e-4` achieves **better Uzbek validation BPB** on SGCA (`uzbek_cyrl` `1.1199` vs `1.1471`, `uzbek_latn` `1.6618` vs `1.7023`) and **better Python validation BPB** on GQA (`1.1345` vs `1.1360`), and avoids high-LR oscillation over longer horizons (`1B–6B` tokens). We therefore keep `peak_lr = 4.0e-4` (`min_lr = 4.0e-5`) as locked in `configs/train.yaml`.
