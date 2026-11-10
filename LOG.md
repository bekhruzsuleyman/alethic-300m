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
- `[git]` 19:20 `cff337f` `▁` escape · 19:20 `52f9ec7` wording · 19:21 `5e7db2a` specials decision ·
  `ac9be0f` LOG.md and AGENTS.md — all pushed to `main` (owner asked).

### Cost

- `[cost]` $0. No servers rented; nothing is running or billing.

### Next

- `[next]` M1, tokenizer on the Hetzner CCX53 ($1.0098 per started hour; estimate $2–3). Needs the
  owner to accept StarCoderData's terms on Hugging Face, rent the server and set `HF_TOKEN` there.
  Not started until the owner says go.

### Shift to Modal, then Vast.ai (20:03)

- `[decision]` Owner: run on **Modal** while the Starter plan's **$30/month free credits** last, then
  move to **Vast.ai**. Hetzner is dropped. This replaces the plan's "Where it runs" row.
- `[finding]` Modal prices and limits (modal.com/pricing and docs, read 2026-11-09):

  | Item | Price | Per hour |
  | --- | --- | --- |
  | H100 SXM | $0.001097/s | $3.949 |
  | A100 80 GB | $0.000694/s | $2.498 |
  | CPU, per physical core (= 2 vCPU) | $0.0000131/s | $0.0472 |
  | Memory | $0.00000222/GiB/s | $0.0080 per GiB |
  | Volume storage | $0.09/GiB/month | — |

  Starter: $30/month free credits, 100 containers, 10 GPUs at once, pay-as-you-go beyond that.
  Max 24 h per function call (default 5 min). GPU functions can be preempted (interrupt signal,
  grace period not documented, then restart on the same input). Volume writes are seen by other
  containers only after a commit plus reload; atomic rename and fsync on Volumes are not documented.
- `[finding]` H100 is cheaper per FLOP than A100 on Modal: $3.949 / 989 TFLOP/s vs $2.498 / 312 TFLOP/s
  → $0.0040 vs $0.0080 per (PFLOP/s · h). H100 stays.
- `[design]` Portability: the `alethic` package stays plain Python (no `modal` import). Modal is one
  launcher, `deploy/modal_app.py`: image from `requirements.txt` on Python 3.12, Volume `alethic-data`
  (v2) at `/data` = `ALETHIC_DATA_DIR`, functions for tests, a Volume probe and H100 training
  (`--resume auto`, 24 h timeout, 2 retries, `--max-cost` required). Vast.ai later runs the same CLI.
- `[design]` Training price in `configs/train.yaml`: H100 3.949 + 4 cores × 0.0472 + 32 GiB × 0.0080 =
  **$4.39/h** (was a $3.00 placeholder).
- `[setup]` `modal` 1.6.3 installed in the laptop venv (an optional `deploy` extra in pyproject).
  Not logged in yet (`python -m modal setup` needs the owner's browser).
- `[finding]` Projected costs on Modal at an assumed 30% MFU (M4 measures the real number).
  Throughput = 0.30 × 989e12 / 3.025e9 = 98,080 tokens/s = 0.668 s per 65,536-token step.

  | Milestone | Work | Estimate |
  | --- | --- | --- |
  | M1 tokenizer | 16 cores + 128 GiB = $1.78/h, about 1–1.5 h | $1.8–2.7 |
  | M2 speed test | 4 cores + 16 GiB = $0.32/h, under 1 h | about $0.3 |
  | M3a 1B subset | measured in M2 | about $2 |
  | M4 smoke | 2 × 2,000 steps × 0.668 s = 0.74 h + about 0.25 h startup, × $4.39 | about $4.4 |
  | M5 LR sweep | 600M tokens / 98,080 = 1.70 h + 0.5 h startup, × $4.39 | about $9.7 |
  | **M1–M5 total** | | **about $18–19 of the $30** |
  | M6 1B pair | 2e9 / 98,080 = 5.67 h × $4.39 | about $25 (does not fit; Vast.ai) |
  | M7 6B pair | 1.2e10 / 98,080 = 34.0 h × $4.39 | about $149 on Modal; Vast.ai price TBD |

- `[next]` Owner: run `python -m modal setup`; set a spending limit in Modal's settings so nothing is
  charged past the free credits; create a Modal Secret named `huggingface` with `HF_TOKEN` (needed
  from M1). Then approve the two smoke checks (`tests`, `probe`, about $0.02 together).
- `[next]` Open question for the Vast.ai move: how the shards get from the Modal Volume to Vast.ai
  (about 12 GB of shards plus raw data). Owner approves the transfer when the time comes.

### Hugging Face repo (20:15)

- `[setup]` Owner named the Hugging Face repo **`bekhruzsuleyman/alethic-300m`**. Checked with
  `huggingface_hub`: it is a **model** repo, **private**, not gated, commit `aa4dfc69`, files
  `.gitattributes` (1,519 B) and `README.md` (21 B). No dataset or Space of that name exists.
  The laptop is logged in to Hugging Face as `bekhruzsuleyman`.
- `[next]` Owner to confirm what goes there. Proposed: the frozen tokenizer (M1), checkpoints and final
  weights (M4–M7), which also solves the Modal → Vast.ai move. Shards (about 12 GB) fit better in a
  separate private dataset repo. Each upload is approved by the owner first.

### Modal login (20:30)

- `[setup]` Owner ran `python -m modal setup` (needed `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`: without it the
  Windows console crashed with a `'charmap' codec` error; bash also strips backslashes, so use
  `.venv/Scripts/python`). Token saved to `~/.modal.toml`, workspace `bekhruzsuleyman`.
- `[setup]` Modal Secret `huggingface` exists (created 20:27 by the owner). `probe_write` now requires the
  key `HF_TOKEN` and reports only whether it is present. No Modal apps running; $0 spent.

### Owner: "don't waste credits on tests; run the real workload; checkpoints must not bottleneck" (20:40)

- `[decision]` No separate test runs on Modal; tests run free on the laptop. The real job is the check.
- `[design]` **Async checkpoints.** A 302M checkpoint is 1.21 GB weights + 2.42 GB Adam moments
  = 3.6 GB. Before: the GPU waited for serialise + SHA-256 + write (about 5–10 s every 500 steps
  = 5.6 min at 30% MFU). Now: snapshot to CPU memory (about 1 s), then a background thread
  serialises, hashes, renames atomically and prunes while training continues; the next save (and
  the program exit) first waits for it. Final, max-cost and signal saves are blocking.
- `[design]` **No per-step GPU sync.** The loss was read with `float()` every step; now the step losses
  stay on the GPU and are read once per log interval (100 steps).
- `[finding]` Free metadata checks from the laptop (2026-11-09):
  - Pinned revisions: starcoderdata `9fc30b57`, fineweb-edu `87f09149`, open-web-math `fde8ef8d`,
    fineweb-2 `af9c1333`, humaneval `7dce6050`, mbpp `4bb6404f` (full hashes in data.yaml).
  - StarCoderData: 92 folders. Candidate languages (parquet GB): shell 1.52, javascript 24.91,
    typescript 10.50, c 19.10, cpp 16.42, rust 2.88, go 8.09; python 23.16 in 59 files.
    **Files are gated: 403 "not in the authorized list"** — the owner must accept the terms.
  - FineWeb-Edu has 6 `CC-MAIN-2025-*` configs (05, 08, 13, 18, 21, 26), 50 files each.
    OpenWebMath 114 files, 27.43 GB. FineWeb-2 `uzn_Latn/train` 1 file, 2.04 GB, 1,233,463 rows;
    `uzn_Cyrl/train` 1 file, 1.97 GB, 1,357,811 rows; both also have a small `test` split.
  - Columns: FineWeb-Edu `text`, `id`, `url`, …; OpenWebMath `url`, `text`, `date`, `metadata`;
    FineWeb-2 `text`, `id`, `url`, …, `language_script`; HumanEval `prompt`, `canonical_solution`, …;
    MBPP `text`, `code`, …. Parquet row groups hold about 1,000 rows.
- `[design]` M1 pipeline: `tokenizer-build` = sample → train → eval → freeze, each stage skipped if its
  output exists. Sample streams (each source; other code per language) are saved one by one, so a
  partial run is never redone. Held-out docs (2,000 per source) are taken before sampling and
  never enter the sample. CPU stages use a torch-free Modal image.
- `[finding]` sentencepiece 0.2.3 removed `out_type="immutable_proto"`; `return_type="proto"` gives
  pieces with byte offsets into the original text (used for the indentation rule).
- `[test]` 34 tests pass on the laptop (19.5 s). End-to-end on real data (FineWeb-2 Uzbek `test`
  files, 6 MB sample, 2,000-id vocab): 1.1 min; Uzbek cleaning changed 68,592 Latin and 155 Cyrillic
  characters; round trip 100% on both; freeze correctly blocked (no Python → indentation gate; tiny
  vocab → Uzbek over 3.0 tokens/word).
- `[cost]` Owner approved running. Launched on Modal (detached): `tokenizer_sample` for
  `fineweb_edu, math, uzbek_latn, uzbek_cyrl` on 2 cores + 16 GiB ($0.22/h). Python and other-code
  streams wait for StarCoderData access and the owner's approval of the language list.

### First Modal job failed; StarCoderData access granted; full M1 launched (21:00)

- `[bug]` The first `tokenizer_sample` job failed at container startup: Modal re-imports the launcher
  inside the container as `/root/modal_app.py`, where `REPO / "requirements.txt"` is `/requirements.txt`
  and does not exist. Only the image build had run (cost close to $0). Fix: read requirements.txt
  only when it exists (on the laptop, where images are built).
- `[decision]` Owner: "I approved" — StarCoderData terms accepted (checked: python file 0 opens,
  218,079 rows, 219 row groups; shell file 0: 551,582 rows) and the seven other-code languages
  approved (`approved: true` in data.yaml).
- `[finding]` StarCoderData columns: `max_stars_repo_path`, `max_stars_repo_name`, `max_stars_count`,
  `id`, `content` — matches data.yaml (`content` text, `max_stars_repo_name` group key).
- `[cost]` Launched `modal run --detach deploy/modal_app.py::tokenizer_all`: all 12 sample streams on
  2 cores + 16 GiB ($0.22/h), then merge + SentencePiece (32 threads) + eval + freeze on
  16 cores + 128 GiB ($1.78/h).
- `[bug]` Second startup failure (seconds, cost close to $0): `cli.main` imported the training error
  classes, which import torch; the CPU image has no torch. Fix: those imports are optional in
  `main`. New test `test_no_torch.py` runs the CLI with torch blocked. 35 tests pass. Relaunched.

### M1 run 2 progress (to 21:46)

- `[finding]` Sample stage 3.4 min: 12 streams, 2.0 GB, exact mix (python 733.3 MB, other code 7 × ~47.6 MB,
  FineWeb-Edu 566.7 MB, math 166.7 MB, uzbek_latn 133.3 MB, uzbek_cyrl 66.7 MB). Long lines dropped:
  math 619, fineweb_edu 33, uzbek_latn 13, uzbek_cyrl 6, code 0.
- `[finding]` SentencePiece: 1,927,141,995 chars, alphabet 14,614, 31,309,048 sentences; seed extraction
  6 chunks, 1,014,614 seeds, about 12 min, mostly single-threaded. EM uses 32 threads = 32 physical
  cores billed against a 16-core request (Modal allows request + 16). Memory used about 10–15 GiB of
  128 GiB requested. → Next CPU jobs: hard limit `cpu=(n, n)` and memory near the measured peak.
- `[finding]` EM sizes: 900,691 → 451,187 → 429,110 → 321,789 → 241,251 → 180,904 (at 21:46);
  rounds about 7–10 min each. Still running (detached); eval + freeze follow automatically.

### M1 run 2 killed at 60 min by local CLI timeout; EM checkpoints added and run 3 spawned (22:45)

- `[bug]` `ap-QiIUdFwfij9BGT2dD06Jpl` stopped at 21:52 (exactly 60 min after its 20:52 start), midway
  through EM (`size=180,904`, 5 of 10 EM rounds done). Cause: `tokenizer_all` is a `@app.local_entrypoint`
  that stayed attached in the local shell; when the shell timed out at 3,600 s it sent `SIGINT` to the
  local `modal` client, which sent a cancellation signal to the remote container (`failed to respond to
  cancellation for too long: 30 seconds - killing task`). All 12 sample streams and the merged 2.0 GB
  `work/sample.txt` had already been committed to the Volume by `tok_sample`, so only the 53 min of
  in-memory SentencePiece training was lost.
- `[design]` **SentencePiece EM checkpoints (`checkpoint_vocabs: [300000, 120000]` → `50000`).**
  Verified on `sentencepiece 0.2.3`: `unigram_model_trainer.cc` accepts `seed_sentencepieces_file`
  (tab-separated `<piece>\t<int64_freq>`) and skips the 12-min suffix-array seed extraction plus earlier
  EM rounds when given pre-pruned pieces. Traps found and handled in [`train.py`](file:///C:/Users/Bekhruz/Home/Alethic-300M/alethic/src/alethic/tokenizer/train.py):
  - Frequencies in `seed_sentencepieces_file` must be `int64` (`SimpleAtoi` rejects floats). We scale
    `exp(score)` from `ModelProto` against single-character probability mass so multi-char and
    `required_chars_` frequencies match in scale.
  - Only `NORMAL` pieces with `len(piece) > 1` and no `\t\r\n\0` go into the TSV (single-char pieces,
    including `\t`, are automatically added from `required_chars_` by `LoadSeedSentencePieces`).
  - Every `.model`/`.vocab` write now trains to `.tmp.{model,vocab}`, `fsync`s, and `os.replace`s
    atomically so a killed write never leaves a corrupt `.model`.
  - [`build.py`](file:///C:/Users/Bekhruz/Home/Alethic-300M/alethic/src/alethic/tokenizer/build.py) and [`modal_app.py`](file:///C:/Users/Bekhruz/Home/Alethic-300M/alethic/deploy/modal_app.py) now call `volume.commit()` immediately after each checkpoint
    (`ckpt_300000`, `ckpt_120000`, `model`, `eval`, `freeze`).
- `[design]` Lowered `tok_train` RAM reservation from 128 GiB to 32 GiB (measured peak was 10–15 GiB),
  cutting base container rate from $1.78/h to $1.01/h, and added `Retries(max_retries=2)`.
- `[test]` Added `test_train_tokenizer_checkpoints` (interrupts after `ckpt_2000`, verifies resume
  reuses `ckpt_2000` and finishes `ckpt_1400` → `1024`). **36 passed in 48.1 s** on laptop.
- `[cost]` Spawned `tok_train` via `modal run --detach deploy/modal_app.py::tokenizer_train --spawn`
  (`ap-gBuq0LbRPWeklivtDdYoop`, call `fc-01M70PKR5ZEFW4DKQNBVCG508W`). The local process exited
  immediately, so no local shell timeout can cancel the remote task.
- `[finding]` Checkpoint progress on `alethic-data` Volume:
  - By 23:16 (32 min in): `ckpt_300000.model`, `ckpt_300000.vocab`, and `seeds_from_ckpt_300000.tsv`
    committed to `/data/tokenizer/work/spm_checkpoints/`.
  - By 23:38 (54 min in): `ckpt_120000.model`, `ckpt_120000.vocab`, and `seeds_from_ckpt_120000.tsv`
    committed to `/data/tokenizer/work/spm_checkpoints/`. Now in the final EM pass (`120,000 → 50,000`),
    followed automatically by `eval` and `freeze`.

---

## 2026-11-10 — M1 finished, frozen, and uploaded to Hugging Face; M2/M3a data pipeline built (00:07–01:10)

- `[finding]` `tok_train` (`ap-gBuq0LbRPWeklivtDdYoop`) finished at 00:07 (`train_seconds: 5003.42`,
  `eval_seconds: 63.36`). All 6 sources passed 100% round-trip (`python` 3.60 B/tok, `code_other` 3.19 B/tok,
  `fineweb_edu` 5.20 B/tok, `math` 3.94 B/tok, `uzbek_latn` 3.85 B/tok & 2.17 tok/word, `uzbek_cyrl`
  5.58 B/tok & 2.55 tok/word).
- `[bug]` `evaluate.py::indentation_cases` initially reported `98.28%` / `97.92%` / `95.99%` on 4 / 8 / 12
  spaces because `b < end and e > start` counted the line's first code token whenever SentencePiece merged
  trailing spaces into the word (e.g. `'▁▁▁▁def'` after `'▁▁▁▁▁▁▁▁'`). Fixed to `b >= start and e <= end`
  (pure indentation tokens inside `[start, end)`), giving **100.00%** across all three widths (`65,080/65,080`,
  `77,101/77,101`, `36,122/36,122`).
- `[finding]` Froze `alethic-tokenizer-v1` (`FROZEN.json`):
  - `model_sha256`: `28685edb461ce3baa8b36f08c32cb04c8f6cb2c1c0c79c6e737816ad214a382b`
  - `config_sha256`: `81ca75999f7b48b5d654ed4454087c85b30688f9b4cbed053f934ef8c39c4bf1`
  - `report_sha256`: `a61d310d2596722b4bbfc1f12f3003bb42782bde465a9cbe0d63d314defb0e2a`
  Synced `FROZEN.json`, `report.json`, `tokenizer.yaml`, and `specials.json` to Modal Volume `alethic-data`
  and uploaded `tokenizer/` to private HF repo `bekhruzsuleyman/alethic-300m` (commit `4d3749af9c614e06579c33dc341d50654d77d4d5`).
- `[design]` Built M2/M3/M4 pipeline modules (all ≤ 300 lines):
  - `data/dedup.py`: 8-byte blake2b document hashing, atomic unit hash `.bin`/`.json`, `build_drop_list`
    keeping the first copy in `(source, file, row)` order via `np.unique(..., return_index=True)`.
  - `data/decontam.py`: 13-gram extractor and `DecontamIndex` keyed by `(first_word, last_word)` against
    pinned HumanEval and MBPP parquet files.
  - `data/encode.py` + `data/shards.py`: row-group (and within-row-group row-slice) work units, parallel
    hash and encode pools, stop rule per stream, `state.jsonl` crash recovery, and `shards/manifest.json`.
  - `eval/bpb.py` + `train/loop.py` + `cli.py`: validation BPB evaluation (`2M` tokens/source every
    `eval_every=1000` steps and full validation at the end of a run, preserving RNG state), plus
    `on_checkpoint` hooks in `AsyncSaver` so every 500-step GPU checkpoint commits to the Modal Volume.
- `[test]` Added `tests/test_data_pipeline.py` (dedup, 13-gram decontam, end-to-end local `data-build`,
  `state.jsonl` resume idempotence, and `evaluate_bpb`). **39 passed, 0 failed** on laptop CPU ($0).

### M2 & M3a 1B-token shard build on Modal CPU (01:10–01:15)

- `[cost]` Spawned `data_build_remote(target_scale=0.1667, workers=16)` (`ap-KWArELTgrWC4n6lyUeHs4x`,
  call `fc-01M70YZ4CY7AYYZC5JE2STRNBK`) on 16 CPU cores + 32 GiB RAM (`$1.01/h`).
- `[finding]` Finished in **183.05 s (3.05 min)**, cost **`$0.05`**:
  - Encoded **`1,061,655,074` tokens** (`1,060,637,876` train + `1,017,198` val across 235 shards).
  - Throughput: **`5,799,943.0` tokens/s total** (`362,496.4` tokens/s/worker — 7.25× the 50k M2 gate).
  - Mix shares (`shards/manifest.json`): `python` **36.36%**, `code_other` **16.31%**, `fineweb_edu` **28.63%**,
    `math` **8.56%**, `uzbek` **10.14%** (`6.91%` Latin + `3.23%` Cyrillic) — max deviation **0.36 pp** from
    the `37 / 17 / 28 / 8 / 10` target (gate: ±2 pp).
  - Dropped **17** exact duplicates out of `1,096,476` documents and **58** benchmark-contaminated documents
    (57 Python, 1 math); normalized `3,590,220` Latin and `11,949` Cyrillic Uzbek apostrophes.

### M4 2,000-step GPU smoke test (`m4_sgca` & `m4_gqa` on H100, 01:18–01:59)

- `[cost]` Ran `m4_sgca` (`ap-vv6G62iyVU0jdkTirOSyIp`, `$1.38`) and `m4_gqa` (`ap-ublzMJncsZBNUAvkPMl5l6`,
  `$1.16`) on `NVIDIA H100 80GB HBM3` + 4 CPU cores + 32 GiB RAM (`$4.39/h`) for 2,000 steps (`131,072,000`
  tokens each) with `--resume auto --stop-step 2000 --max-cost 3.0`.
- `[finding]` **Speed & MFU:**
  - `m4_sgca`: **`139,656` tokens/s**, **`42.72% MFU`** (projected 6B run: `11.93 h`).
  - `m4_gqa`: **`148,952` tokens/s**, **`45.57% MFU`** (projected 6B run: `11.19 h`).
- `[finding]` **Loss & validation BPB (`1,017,192` val tokens across all 6 sources):**
  - Step 1,000: SGCA train loss **`4.4792`** vs GQA `4.7455`; SGCA val `overall_bpb` **`1.526906`** vs GQA
    `1.605544` (SGCA ahead on all 6 sources at end of warmup).
  - Step 2,000: SGCA train loss `3.3186` vs GQA `3.3185`; val `overall_bpb`: SGCA `1.173432` vs GQA `1.163813`
    (SGCA ahead on `fineweb_edu` `1.3136` vs `1.3147`, `uzbek_latn` `1.5795` vs `1.5926`, and `uzbek_cyrl`
    `1.0632` vs `1.0758`; GQA ahead on `python` `0.8897` vs `0.9064`, `code_other` `1.0996` vs `1.1428`, and
    `math` `1.4511` vs `1.4701`).
  - Atomic 500-step checkpoints (`step_0000500`, `step_0001000`, `step_0001500`, `step_0002000`, pruned to
    keep the last two: `1500` and `2000`) verified on the Volume and backed up to private HF repo
    `bekhruzsuleyman/alethic-300m` (`shards_manifest: 46dd2842`, `m4_sgca: 0bbf553e`, `m4_gqa: ce0d1219`).

### M5 learning-rate sweep (`100M` tokens × `{2e-4, 4e-4, 8e-4}` × `{sgca, gqa}`, 02:04–03:24)

- `[cost]` Ran all 6 M5 configurations (`1,526` steps = `100,007,936` tokens each, 5% warmup = `76` steps,
  cosine decay to `peak_lr / 10`) in one H100 session (`ap-n4ANFR7D7HvtDgPzVprO9a`, 80 min total, **`$5.48`**).
- `[finding]` **Final step 1,526 validation BPB (all `1,017,192` val tokens):**
  - `2e-4`: **SGCA `1.409033`** (train loss `3.9094`) vs **GQA `1.503421`** (train loss `4.1734`) → **SGCA wins by `-0.0944` BPB**.
  - `4e-4`: **SGCA `1.269719`** (train loss `3.4892`) vs **GQA `1.394150`** (train loss `3.8329`) → **SGCA wins by `-0.1244` BPB**.
  - `8e-4`: **SGCA `1.244308`** (train loss `3.3990`) vs **GQA `1.378295`** (train loss `3.7995`) → **SGCA wins by `-0.1340` BPB**.
  - Across all 3 learning rates and all 6 data sources (`python`, `code_other`, `fineweb_edu`, `math`,
    `uzbek_latn`, `uzbek_cyrl`), **SGCA outperforms GQA at the end of cosine decay**.
  - `4e-4` gives SGCA its best Uzbek BPB (`1.1199` Cyrillic, `1.6618` Latin vs `1.1471` / `1.7023` at `8e-4`)
    and GQA its best Python BPB (`1.1345` vs `1.1360` at `8e-4`), confirming `configs/train.yaml`'s
    `peak_lr: 4.0e-4` (`min_lr: 4.0e-5`) for M6.
  - Backed up all 6 M5 runs to `bekhruzsuleyman/alethic-300m` (`cb189284`, `65210f95`, `4a8ff9b9`,
    `5c571e1d`, `f3bb75ce`, `baf0aee7`).

### M6 1B-token matched pair (`m6_sgca` & `m6_gqa`, `15,259` steps each on H100, 03:29–07:30)

- `[cost]` Spawned `m6_pair_remote(max_cost_per_run=10.5)` (`ap-YlOc7OTvBjhEnwoQ0zjbJA`, call
  `fc-01M716W0JBKJN7NB5S13SCSE7K`) at 03:29 on `NVIDIA H100 80GB HBM3` (`$4.39/h`).
  Both models trained for `15,259` steps (`1,000,013,824` tokens = `3.025 × 10¹⁸` FLOPs each) with
  identical seed (`1234`), data order, `peak_lr = 4.0e-4`, `min_lr = 4.0e-5`, and 5% linear warmup
  (`763` steps), evaluating validation BPB every 1,000 steps and saving Volume-committed checkpoints
  every 500 steps.
- `[test]` **Intentional stop & resume gate on `m6_sgca`:**
  - `m6_sgca` ran `0 → 1,000` steps with `--stop-step 1000`, saved and committed `step_0001000` to the
    Modal Volume, and then resumed (`--resume auto`) cleanly from `step_0001000` to `step_0015259`
    (`runs/m6_sgca/runs.jsonl` records `start_step: 0` and `start_step: 1000` with identical config,
    shard, and tokenizer SHA-256 hashes).
- `[finding]` **`m6_sgca` completed at 05:34 (`2.07 hours`, `$9.07`):**
  - Sustained speed: **`138,882 – 139,316` tokens/s**, **`420.2 – 421.5 TFLOP/s`**, **`42.48% – 42.62% MFU`**.
  - Final step `15,259` train loss: **`2.3579`** nats.
  - Final step `15,259` full validation (`1,017,192` tokens = `4,085,874` UTF-8 bytes):
    - `overall_bpb`: **`0.847203`**
    - `python`: **`0.607403`** BPB (`1.457970` nats)
    - `code_other`: **`0.695311`** BPB (`1.389513` nats)
    - `uzbek_cyrl`: **`0.799914`** BPB (`3.111462` nats)
    - `fineweb_edu`: **`1.013832`** BPB (`3.625675` nats)
    - `math`: **`1.094650`** BPB (`3.095357` nats)
    - `uzbek_latn`: **`1.202895`** BPB (`3.261190` nats)
  - Final checkpoint `runs/m6_sgca/checkpoints/step_0015259`:
    `state_sha256 = d997cb8d920adf8f2e217d24cca65d41944c7fc60cbd63841bf3457a6a58985c`.
  - Automatically uploaded `runs/m6_sgca` (logs + `step_0015259`) to Hugging Face `bekhruzsuleyman/alethic-300m`
    (commit `748494ce5b07ceca699b86c5ae9f98589930fc5f`) before starting `m6_gqa`.
- `[finding]` **`m6_gqa` completed at 07:30 (`1.91 hours`, `$8.36`):**
  - Sustained speed: **`148,940 – 149,677` tokens/s** (`+6.7%` faster than SGCA), **`450.6 – 452.8 TFLOP/s`**,
    **`45.56% – 45.79% MFU`** (`+3.1 pp` higher MFU than SGCA).
  - Final step `15,259` train loss: **`2.2490`** nats (`-0.1088` nats lower than `m6_sgca`).
  - Final step `15,259` full validation (`1,017,192` tokens = `4,085,874` UTF-8 bytes):
    - `overall_bpb`: **`0.806841`** (`-0.040362` BPB / `-4.8%` lower than `m6_sgca`)
    - `python`: **`0.569646`** BPB (`1.367341` nats; `-0.0378` BPB vs SGCA)
    - `code_other`: **`0.646453`** BPB (`1.291876` nats; `-0.0489` BPB vs SGCA)
    - `uzbek_cyrl`: **`0.761652`** BPB (`2.962633` nats; `-0.0383` BPB vs SGCA)
    - `fineweb_edu`: **`0.976563`** BPB (`3.492396` nats; `-0.0373` BPB vs SGCA)
    - `math`: **`1.049375`** BPB (`2.967331` nats; `-0.0453` BPB vs SGCA)
    - `uzbek_latn`: **`1.154667`** BPB (`3.130439` nats; `-0.0482` BPB vs SGCA)
  - Final checkpoint `runs/m6_gqa/checkpoints/step_0015259`:
    `state_sha256 = 8a8567e55440b221918b93d5fab0b9f565b0dd59ae21699e63aab7297b795a68`.
  - Automatically uploaded `runs/m6_gqa` (logs + `step_0015259`) to Hugging Face `bekhruzsuleyman/alethic-300m`
    (commit `3c900c7e72aef7e3a5c1f8e7554d1d86261d7318`).
- `[finding]` **Why SGCA won M5 (100M tokens) but GQA won M6 (1B tokens) — the `step 2,200` crossover:**
  - Both models are strictly Iso-Parameter (`302,908,416` params) and strictly Iso-FLOP (`3.0254 GFLOP/tok`,
    since SGCA's $1024 \times 1024$ `gate_proj` adds `6.29 MFLOP/tok/layer` while shrinking `d_ff` from
    `3072 → 2730` removes `6.30 MFLOP/tok/layer`).
  - **Early / low-FLOP regime (`steps 100 → 2,200`, `0 → 144M` tokens, `< 0.44 EFLOPs`):** SGCA's
    sinusoidal gate (`C = (W·H + b) ⊙ (alpha ⊙ sin(A))`) accelerates early representation learning, giving
    much lower train loss at step 1,000 (`4.2858` vs `4.7187`) and step 1,500 (`3.5302` vs `3.6608`), and
    lower validation BPB at step 1,000 (`1.4627` vs `1.6024`) and step 2,000 (`1.1564` vs `1.1619`).
  - **Exact crossover at step `2,200` (`144.2M` tokens, `0.44 EFLOPs`):** Both models hit `loss = 3.245` nats.
  - **Scaling regime (`steps 2,300 → 15,259`, `150M → 1.0B` tokens, `0.45 → 3.03 EFLOPs`):** Once past
    the initial warmup/low-compute phase, spending `1,050,624` parameters per layer on SwiGLU FFN width
    (`d_ff = 3072` in GQA vs `2730` in SGCA) provides higher capacity per FLOP than spending them on
    `gate_proj`. By step `10,500` (`688M` tokens), `m6_gqa` already surpasses `m6_sgca`'s final step-`15,259`
    (`1.0B`-token) validation BPB across all 6 sources, finishing `15,259` steps ahead by **`-0.0404` BPB**
    and **`-0.1088` nats** while also running **`6.7%` faster** on H100 (`45.6%` vs `42.5%` MFU).
- `[finding]` **Host CPU & memory right-sizing (`deploy/modal_app.py`):**
  - Modal metrics during M6 showed GPU containers used only **`1.29` CPU cores** (of `4.0` requested) and
    **`7.25 GiB` host RAM** (of `32.0 GiB` requested) because `ShardDataset` uses zero-copy `np.memmap`.
  - Updated `train_remote` and `m6_pair_remote` in `deploy/modal_app.py` to `cpu=2, memory=12288` (`12 GiB`),
    cutting host CPU/RAM overhead from `$0.44/h` to `$0.19/h` and lowering the H100 container rate from
    `$4.39/h` to **`$4.14/h`** for any future GPU runs.
- `[cost]` **Total Modal credits spent across M1–M6:**
  - M1: `~$3.40` · M2/M3a: `$0.05` · M4: `$2.54` · M5: `$5.48` · M6: `$17.43` (`$9.07` SGCA + `$8.36` GQA)
  - **Cumulative total:** **`~$28.90`** of the `$30.00` monthly free Starter credit (`~$1.10` remaining).
  - **Active containers (`modal app list`):** **0 running** (`ap-YlOc7OTvBjhEnwoQ0zjbJA` stopped at 07:30).
    Nothing is running or billing.
- `[next]` All milestones through M6 are complete, verified, and backed up to Hugging Face (`bekhruzsuleyman/alethic-300m`).
  Stop and wait for the owner's decision on whether/where to run M3b (6B-token shards) and M7 (6B-token training).

