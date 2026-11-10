# AGENTS.md — read this first

Guide for any AI agent working on this repo. Read it fully, then `LOG.md` (the whole story so far)
and `docs/RUN_RECORD.md` (formal record per milestone).

## What this project is

Alethic-300M: a from-scratch rewrite that tokenizes 6B tokens of code-first data (10% Uzbek) and
trains two matched 300M-parameter models, **SGCA** and a **GQA** baseline, to compare them fairly.
The owner is a student on a tight budget (about $20 now, about $12 more per week). Every stage must
be cheap, fast and restartable.

The source of truth is the owner's plan, "Alethic-300M rewrite: implementation plan for an AI"
(2026-11-09). It is **not in this repo**; on the owner's laptop it sits one folder up, in
`../Alethic-300M rewrite implementation plan for an AI.md`. If you cannot read it, ask the owner for
it before building anything.

## Current state

| Milestone | Status |
| --- | --- |
| M0 repo skeleton, model, loader, all tests (laptop CPU) | **Done**: 39/39 tests pass |
| M1 tokenizer on Modal CPU | **Done**: 50,000-id Unigram trained, all gates passed (100% roundtrip, 100% indentation, Uzbek 2.17/2.55 tok/word), frozen (`28685edb...`), uploaded to HF (`4d3749af`) |
| M2 & M3a data speed test + 1B-token subset on Modal CPU | **Done**: `1,061,655,074` tokens in 183s (`362.5k` tok/s/worker, `5.80M` tok/s total, `$0.05`), mix within ±0.36 pp of `37/17/28/8/10` (Uzbek `10.14%`), 58 decontam + 17 dedup drops |
| M4 2,000-step GPU smoke test on Modal H100 (`m4_sgca` & `m4_gqa`) | **Done**: SGCA `139.7k` tok/s (`42.72% MFU`, val BPB `1.5269 → 1.1734`), GQA `149.0k` tok/s (`45.57% MFU`, val BPB `1.6055 → 1.1638`); checkpoints & logs backed up to HF |
| M5 LR sweep (`100M` tok × `{2e-4, 4e-4, 8e-4}` × `{sgca, gqa}`) | **Done**: SGCA beats GQA at all 3 LRs and on all 6 sources (`1.4090 vs 1.5034` @ `2e-4`, `1.2697 vs 1.3942` @ `4e-4`, `1.2443 vs 1.3783` @ `8e-4`); `4e-4` confirmed for M6 |
| M6 1B-token matched pair (`15,259` steps each on H100) | **Done**: `m6_sgca` (`overall_bpb = 0.847203`, `loss = 2.3579`, `42.5% MFU`, HF `748494ce`) vs `m6_gqa` (`overall_bpb = 0.806841`, `loss = 2.2490`, `45.6% MFU`, HF `3c900c7e`); SGCA leads `< 144M` tok, GQA leads `> 144M` tok (`-0.0404` BPB at 1B) |
| Run B: GQA-narrow ablation (`abl_gqa_narrow`, 6,000 steps, H100) | **Done**: B6 = `0.9302` vs SGCA `0.9612` / GQA `0.9273` → f = `0.086` (< 0.33: the **gate** is the main cost, not the FFN cut); `150.8k` tok/s, `$3.14`; checkpoint on Volume only |
| M3b / M7 / M8 | Wait for owner after M6 (`~$28.90 / $30.00` Modal free credits used; `0` containers running) |

Update this table when a milestone's gate is passed.

## Hard rules (from the plan and the owner)

1. **One milestone at a time.** Build it, run its tests, stop at its gate, report, and **wait for the
   owner** before the next milestone.
2. **Ask before you:** spend money (rent a server, start a GPU), upload anything (data, checkpoints,
   `git push`), choose the other-code languages, or change any decision below.
3. **Never** print, log or commit `HF_TOKEN` or any credential. The pipeline writes only to the
   machine's own disk (on Modal: the private Volume `alethic-data`). Moving data between providers
   (Modal → Vast.ai) is an upload: the owner approves it first.
4. **Never relax a gate** to make it pass. If a gate fails, report the cause and the numbers.
5. **Verify, don't assume.** Check the installed SentencePiece version for every option, and the real
   subset and column names of every dataset. Report what you find.
6. **Fail loudly.** Placeholders (`REPLACE_`, `YOUR_`), unpinned revisions (`revision: null` at build
   time) and a tokenizer hash mismatch must stop the program with a message saying what to do.
7. **Keep it small and readable.** Each file under 300 lines, type hints on public functions, plain
   comments on every function (the owner learns by reading the code). Dependencies are only: torch,
   numpy, pyarrow, sentencepiece, regex, huggingface_hub, pyyaml, pytest. `modal` is allowed only in
   `deploy/modal_app.py` (laptop launcher); the `alethic` package must never import it.
8. **Be deterministic.** Seeds live in the configs; same inputs give the same outputs.
9. **Record everything.** Add to `LOG.md` as you work (see its conventions) and add a section to
   `docs/RUN_RECORD.md` at every milestone: commands, config and tokenizer hashes, measured speeds,
   costs, surprises.
10. **At every stop, say what is still running or billing**: Modal apps/containers (check
    `modal app list`) and, later, Vast.ai instances. Vast.ai bills storage while an instance exists,
    even when stopped; remind the owner to download results first, then destroy it. Report Modal
    credits spent against the $30/month free allowance.
11. **Every GPU run gets `--max-cost`.** The Modal launcher refuses to start training without it.
12. **Reports use numbers and worked math**, not long prose. The owner prefers that.

## Locked decisions

Owner decisions are final. "Default" rows were set by the plan's author and may be changed only with
the owner's approval.

| Topic | Decision |
| --- | --- |
| Tokenizer | SentencePiece Unigram, 50,000 ids, byte fallback, 64 reserved slots `<sp00>`–`<sp63>` |
| Id layout | 0–3 `<unk> <s> </s> <pad>`, 4–67 `<spNN>`, 68–323 bytes, learned pieces from 324; EOS = 2 |
| Normalization | Identity (no NFKC) **plus the U+2581 escape** (owner, 2026-11-09): real `▁` ↔ U+E000 `0`, real U+E000 ↔ U+E000 `1`, stored inside the `.model`. `decode(encode(x)) == x` for 100% of text |
| Special slots | Stay `user_defined_symbols`, as the plan says (owner, 2026-11-09); typed `<spNN>` text can produce the id |
| Data | 6B tokens: Python 2.2B, other code 1.0B, FineWeb-Edu 1.7B, OpenWebMath 0.5B, Uzbek 0.6B (Latin 0.4B, Cyrillic 0.2B) |
| Models | SGCA and GQA, exactly 302,908,416 parameters each, identical data and settings |
| Shape | 24 layers, d_model 1,024, 16 query / 4 KV heads, head dim 64, context 4,096, RoPE θ 10,000, tied embeddings, SwiGLU 2,560 (GQA) / 2,218 (SGCA) |
| SGCA | `C = (W·H + b) ⊙ (alpha ⊙ sin(A))`, `A = o_proj(GQA(H))`, `X' = X + C`; alpha is a vector of length 1,024 starting at 1.0 |
| Training | 65,536 tokens per step, 91,553 steps, AdamW (0.9, 0.95), wd 0.1 on matrices only, LR 4e-4 peak, 1,000 warmup, cosine to 4e-5, clip 1.0, bf16 + torch.compile, no activation checkpointing |
| Data order | One pass, global seeded shuffle of 4,096-token windows, no cross-document masking |
| Machines | **Changed by the owner on 2026-11-09:** Modal (CPU and H100) while the $30/month free Starter credits last, then Vast.ai. Laptop for tests and launching only. Hetzner is no longer used |

## Commands

```text
python -m alethic tokenizer-build | data-fetch | data-build | train | eval
python -m alethic train --model configs/model_sgca.yaml --train configs/train.yaml --resume auto --max-cost 40
python -m pytest -v

# Modal (from the repo root, after `python -m modal setup`; each call spends credits)
modal run deploy/modal_app.py::tests      # test suite on Linux, about $0.01
modal run deploy/modal_app.py::probe      # Volume rename/fsync/commit check, < $0.01
modal run deploy/modal_app.py::train --model sgca --max-cost 5 --stop-step 2000
modal run --detach deploy/modal_app.py::tokenizer_sample --streams fineweb_edu,math   # M1 streams, 2 cores
modal run --detach deploy/modal_app.py::tokenizer_train                              # M1 merge+train+eval+freeze, 16 cores
```

On Windows prefix Modal commands with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` and call
`.venv/Scripts/python -m modal ...` (bash strips backslashes).

On Modal, `ALETHIC_DATA_DIR=/data` is the Volume `alethic-data` (v2). On Vast.ai, clone the repo,
`pip install -r requirements.txt -e .`, set `ALETHIC_DATA_DIR`, and run the same CLI.

`train` and `tokenizer-build` are implemented; the others print which milestone builds them. Large files go
under `$ALETHIC_DATA_DIR`; paths in the configs are relative to it.

## Repo map

| Path | What it holds |
| --- | --- |
| `configs/` | `tokenizer.yaml`, `data.yaml`, `model_sgca.yaml`, `model_gqa.yaml`, `train.yaml` |
| `src/alethic/config.py` | YAML loading, placeholder check, `ALETHIC_DATA_DIR`, hashing, atomic JSON |
| `src/alethic/tokenizer/` | `train.py` (options + `▁` escape), `codec.py`, `specials.py`, `freeze.py`, `sample.py` (resumable weighted sample), `evaluate.py` (gates), `build.py` (stages) |
| `src/alethic/data/` | `shards.py` (crash-safe shards, `state.jsonl`), `loader.py`, `uzbek.py`, `encode.py` (`is_validation`), `fetch.py` (pinned downloads, document reader); `dedup`/`decontam` are M2 stubs |
| `src/alethic/model/` | `layers.py`, `attention_gqa.py`, `attention_sgca.py`, `transformer.py`, `params.py` |
| `src/alethic/train/` | `loop.py`, `optim.py`, `checkpoint.py`, `cost.py` |
| `src/alethic/eval/` | `bpb.py` (formula only so far), `humaneval.py` (stub) |
| `deploy/modal_app.py` | Modal launcher: image, Volume, tests, Volume probe, H100 training with retries |
| `tests/` | one file per area; `conftest.py` trains a tiny 1,024-id tokenizer with the real settings |

## Traps already found (don't fall into them again)

- **`▁` (U+2581) is SentencePiece's space marker.** Without the escape a real `▁` decodes as a space.
  `escape_whitespaces=false` is rejected by the 0.2.3 Unigram trainer. Keep `u2581_escape: true`;
  the codec refuses a model without the escape.
- **Whitespace `user_defined_symbols` must be written with `▁`, not spaces.** Space versions are
  accepted but never match; `train.py` refuses them.
- **Writing `\u2581`, `\ue000`, `\t` escapes through shell heredocs or sed** can turn the escapes into raw characters
  or real tabs and newlines. Edit those lines with an exact-match file edit and check the bytes afterwards.
- **The laptop** runs Windows with 3.65 GB RAM (often under 0.5 GB free). Don't build the full 302M model
  there; use `torch.device("meta")` for shape and count checks. Use the venv at `.venv`
  (Python 3.12.10). Set `PYTHONIOENCODING=utf-8` when printing pieces to the Windows console.
- **pytest temp folder:** if the default temp folder is blocked (sandboxes), pass `--basetemp <dir>`.
- **`.gitignore`** uses `/data/` (root only). A bare `data/` would hide `src/alethic/data/`.
- **Modal specifics** (docs, 2026-11-09): writes reach other containers only after a commit (background
  commits every few seconds, `volume.commit()`, final commit at shutdown) and a reader must
  `volume.reload()`. Many containers may write *different* files at once; never let two write the same
  file (so a parallel data build needs one coordinator writing `state.jsonl`). Max 24 h per call;
  GPU functions can be preempted (an interrupt signal, then a restart of the same input). Atomic
  rename on Volumes is not documented: run the `probe` before trusting crash safety there.
- **Checkpoints are written in a background thread** (`AsyncSaver`). Never read model state while a
  save runs except through `to_cpu` snapshots; always `saver.wait()` before exit.
- **Don't call `.item()` / `float()` on GPU tensors every step**; it stalls the GPU. Read at log steps.
- **SIGTERM** is how rented GPUs vanish; the loop saves a checkpoint on SIGTERM/SIGINT and every 500 steps.

## Hugging Face

Model repo **`bekhruzsuleyman/alethic-300m`** (private; on 2026-11-09 it held only `.gitattributes` and
`README.md`). Every upload to it needs the owner's approval first, file by file or batch by batch;
the pipeline never uploads by itself. StarCoderData's license requires derived data to stay private,
so never make a repo holding data or models trained on it public without the owner's explicit decision.

## Git

Remote: https://github.com/bekhruzsuleyman/alethic-300m (`main`). Commit freely, but **push only when
the owner asks**. End commit messages with the co-author line your harness requires.
