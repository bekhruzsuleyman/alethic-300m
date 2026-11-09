# Alethic-300M research log

A running, dated record of everything done on this project: setup, design choices, experiments,
bugs, owner decisions and measured numbers. It is written to be turned into documentation later.
`docs/RUN_RECORD.md` is the formal per-milestone record the plan asks for; this log keeps the
whole story, including dead ends.

**Conventions.** Newest day at the bottom. Each entry has a tag:
`[setup]` machine and software, `[design]` a choice the plan left open, `[finding]` something
measured or verified, `[bug]` a defect found and fixed, `[decision]` an owner decision,
`[test]` test results, `[git]` commits and pushes, `[cost]` money spent, `[next]` what comes next.
Numbers are exact where they were measured. Times are local (UTC+5).

---

## 2026-11-09 — M0: repo skeleton, model, loader, all tests (laptop CPU)

### Starting point

- `[setup]` Plan: "Alethic-300M rewrite: implementation plan" (2026-11-09). M0 builds the repo
  skeleton, both models, the data loader and every test in the plan's table, on the laptop CPU, at $0.
- `[setup]` Laptop: Windows 11 Home, 8 logical CPUs, 3.65 GB RAM (about 0.46 GB free during work),
  about 26 GB free disk. Installed Pythons were only 3.14.7 and 3.15.0; the plan names 3.12.
- `[setup]` Installed Python 3.12.10 with `py install 3.12` (free, from python.org) and a venv at
  `alethic/.venv`. Packages (CPU build of torch):

  | Package | Version |
  | --- | --- |
  | torch | 2.15.0+cpu |
  | sentencepiece | 0.2.3 |
  | numpy | 2.5.4 |
  | pyarrow | 26.0.1 |
  | regex | 2026.11.9 |
  | huggingface_hub | 2.4.0 |
  | PyYAML | 6.0.3 |
  | pytest | 9.2.0 |

### Design choices the plan left open

- `[design]` **Repo root** is the `alethic/` folder from the plan's layout. One extra module,
  `src/alethic/config.py`, holds YAML loading, the placeholder check (`REPLACE_`, `YOUR_`), the
  `ALETHIC_DATA_DIR` lookup and file hashing, so every stage shares one implementation.
- `[design]` **Later-milestone modules are stubs** (`fetch`, `dedup`, `decontam`, the sample builder,
  tokenizer evaluation, the BPB driver, HumanEval). Each raises `NotImplementedError` naming its milestone.
- `[design]` **Validation split** `is_validation(source, key)` lives in `data/encode.py`. It hashes
  only the key (8-byte blake2b, personalised `alethic-split`), not the source, so a repo that appears
  in both the Python and other-code sources lands on the same side in both. Bucket rule:
  `hash mod 1,000,000 < 1,000` → 0.1%.
- `[design]` **Loader windows.** Window w = tokens `w·4096 … w·4096 + 4096` (4,097 tokens: inputs are
  the first 4,096, targets the last 4,096), so consecutive windows share one token and every token is
  used once as an input. Number of windows = (N − 1) // 4096. Order: one permutation from
  `numpy.random.default_rng([seed, epoch])`. Step k takes permuted windows 16k … 16k + 15. If a run
  asks for more steps than one pass holds, the next epoch uses a new seeded permutation.
- `[design]` **Shards** go to `shards/train/<source>/<name>-<part>.bin` and `shards/val/...`, where
  `<name>` is the input file plus its row-range unit (e.g. `file3-r000`), because one file is split
  into several units that run in parallel. Validation sidecars also store per-document byte and token
  counts (needed for bits per byte).
- `[design]` **Crash safety.** `.bin` written to `.tmp` → flush → fsync → rename, then the `.json`
  sidecar the same way, then one line in `state.jsonl`. A torn last line in `state.jsonl` is cut off on
  restart and that unit is redone; redoing rewrites identical bytes.
- `[design]` **specials.json** = `{"format": 1, "names": [...]}`; the slot is the position in the
  list, so slot s has id 4 + s. Append-only is enforced by comparing against an older copy (the one
  saved next to each checkpoint) and against the file on disk before every save.
- `[design]` **Checkpoints** are folders `runs/<name>/checkpoints/step_0000500/` holding `state.pt`,
  a copy of `specials.json` and `manifest.json` (SHA-256 of both, config hashes, versions). Written as
  `step_….tmp/` and renamed when complete. Resume refuses if the model config, data order (seed,
  sequence length, batch, steps), shard list or tokenizer hash changed.
- `[design]` **Cost guard** counts all wall time since the process started (startup, compile and
  checkpoint saves included, because a rented GPU bills for them), summed across resumes.
- `[design]` Model details fixed in code: RMSNorm computed in float32; RoPE in the "rotate half"
  form; SwiGLU matrices named `gate`, `up`, `down`; init std 0.02, with `o_proj` and `ffn.down` at
  0.02 / √48 ≈ 0.00289; norms and alpha start at 1, gate bias at 0. SGCA subclasses GQA attention,
  so both share the same q/k/v/o code.

### SentencePiece experiments (sentencepiece 0.2.3, tiny 1,024-id model on an 875 KB synthetic corpus)

- `[finding]` **Id layout** with the plan's settings: 0–3 `<unk> <s> </s> <pad>`, 4–67
  `<sp00>`…`<sp63>`, 68–323 byte pieces `<0x00>`…`<0xFF>` (`<0x0A>` = 78), learned pieces from 324.
- `[finding]` **Indentation** pieces were learned without any help: newline + 4, 8, 12 spaces →
  `<0x0A>` + one piece of 4, 8 or 12 `▁` (2 tokens). 16 spaces → 3 tokens (two 8-space pieces).
- `[finding]` **A literal `▁` (U+2581) does not round-trip.** SentencePiece replaces spaces with `▁`
  before tokenizing, so a real `▁` decodes as a space: `'a▁b'` → `['a', '▁b']` → `'a b'`.
  This would break the plan's 100% round-trip gate on any document containing `▁` (common in
  tokenizer-related Python code).
- `[finding]` **`escape_whitespaces=false` is not usable for Unigram.** Training stops with
  `INTERNAL: unigram_model_trainer.cc(828) [normalizer_spec_.escape_whitespaces()]`.
- `[finding]` **Whitespace runs as `user_defined_symbols`** (the plan's fallback): symbols written with
  real spaces are accepted by the trainer but never match, because input spaces become `▁` first.
  Written as `▁▁▁▁…` they work (checked with 16 and 20 `▁`; both one piece, exact round trip).
  `train.py` now refuses the space form with a message.
- `[finding]` **Typed `<spNN>` text** encodes to the special id (`'<sp00>'` → `['<sp00>']`), because
  the slots are `user_defined_symbols`. Other angle-bracket text stays ordinary:
  `'<thinking>'` → `['<0x3C>', 't', 'h', 'in', 'k', 'in', 'g', '<0x3E>']`.
- `[finding]` Tiny-tokenizer training time: 4.8–6.2 s with 4 threads (10.3 s with 2 extra symbols).
  These numbers do not predict the 2 GB M1 run.

### The U+2581 escape

- `[decision]` Owner: "100% of documents should be tokenized" exactly; asked whether special codes can
  stand in for `▁`.
- `[design]` Escape built into the `.model` file as two normalization rules plus the two reverse
  rules, applied by SentencePiece itself:

  | Real text | Encoded as | Decoded back to |
  | --- | --- | --- |
  | U+2581 `▁` | U+E000 + `0` | U+2581 |
  | U+E000 | U+E000 + `1` | U+E000 |

  U+E000 is the first private-use character. Escaping the marker itself makes the scheme lossless,
  like `\\` for a backslash. Everything else is identity (no NFKC).
- `[finding]` Verification: 0 failures in 5,000 random strings built from `▁`, U+E000, `0`, `1`,
  space, newline and arbitrary code points; NFKC-sensitive text (`ﬁ ① é ｶ`) unchanged; indentation
  still 2 tokens. A real `▁` costs 4 tokens (`<0xEE> <0x80> <0x80> 0`).
- `[finding]` Caveat: llama.cpp and Ollama ignore SentencePiece normalization rules, so an app running
  the model there must apply the same two swaps itself. Training data and the Python codec are exact.
- `[design]` The codec refuses to load a model that lacks the escape (it round-trips `▁`, U+E000 and
  `a▁ U+E000 0` at load time).

### Bugs found by the tests

- `[bug]` `train` through the CLI crashed at its first checkpoint: nothing had created
  `specials.json`, so copying it failed. Fix: the trainer loads the registry at start (creating it with
  `image`, `img_start`, `img_end` if missing).
- `[bug]` When `--max-cost` stopped training between log steps, no row was written to
  `train_log.jsonl`, so the last numbers were lost. Fix: the stop reason is decided before logging and
  a stop always writes its row.
- `[bug]` (found in review, not by a test) The cost guard counted only time inside training steps,
  which would under-report rental cost. Fix: count wall time since process start.
- `[bug]` (found in review) The resume check used only shard file names, which can repeat across
  sources. Fix: include the source folder.
- `[test]` The kill test originally only checked that kills happened at all. Tightened: every one of
  the 3 outside kills must land before the build finishes. It does.

### Test results

- `[test]` First full run: 28 passed, 1 failed (the specials bug). Second run: 28 passed, 1 failed
  (the log-row bug), plus 1 expected failure documenting `▁`.
- `[test]` After the `▁` escape: **30 passed, 0 failed, 30.7 s** (earlier full runs: 40–54 s; the plan's
  limit is 5 min).
- `[test]` Numbers the tests check:
  - Parameters, both models: 24 × 10,487,808 + 51,200,000 + 1,024 = **302,908,416**
    (SGCA adds 1,048,576 + 1,024 + 1,024 = 1,050,624 per layer; its FFN at 2,218 removes exactly
    (2,560 − 2,218) × 3 × 1,024 = 1,050,624).
  - FLOPs per token: 6 × 302,908,416 + 12 × 24 × 1,024 × 4,096 = **3,025,410,048**.
  - Steps for 6B tokens: ceil(6,000,000,000 / 65,536) = **91,553**.
  - Resume: 20 straight steps vs 10 + resume + 10, max difference ≤ 1e-6 (both models).
  - Overfit: loss < 0.1 on one repeated batch within 200 steps (both models).
  - Split: 0.1% ± 0.05% of 200,000 keys in validation.
  - Crash safety: one crash halfway through a shard write plus 3 outside kills; every rerun gave
    byte-identical shards.
- `[finding]` The full 302M model was not run on the laptop: it needs at least 1.2 GB of RAM for the
  weights alone and only about 0.46 GB was free. Shapes and counts are checked on PyTorch's meta
  device instead. First real speed numbers come in M4 on the H100.
- `[setup]` In the assistant's sandbox, pytest's default temp folder was blocked, so tests ran with
  `--basetemp`; a normal shell does not need this.

### Decisions

- `[decision]` 100% exact round trip, including `▁`: implemented with the escape above. This turns the
  "Text normalization: None" default into "none, plus the `▁` escape".
- `[decision]` Special slots stay `user_defined_symbols`, as the plan says. Typed `<spNN>` text can
  produce a special id; the serving app should handle user text containing `<sp` once a slot is used
  (first planned use: M8 image tokens).
- `[decision]` Keep this log (`LOG.md`) for the whole project, to write the research website later.

### Git

- `[git]` Repo: https://github.com/bekhruzsuleyman/alethic-300m (`main`).
- `[git]` 19:11 `571300c` M0: repo skeleton, model, loader and all tests — **pushed** (owner asked).
- `[git]` 19:20 `cff337f` `▁` escape · 19:20 `52f9ec7` wording · 19:21 `5e7db2a` specials decision —
  committed locally, not pushed yet.

### Cost

- `[cost]` $0. No servers rented; nothing is running or billing.

### Next

- `[next]` M1, tokenizer on the Hetzner CCX53 ($1.0098 per started hour; estimate $2–3). Needs the
  owner to accept StarCoderData's terms on Hugging Face, rent the server and set `HF_TOKEN` there.
  Not started until the owner says go.
