"""Phase 3 for Alethic-300M GQA: continue the phase-2 model on the rest of the unseen data (second Modal workspace).

Same shape as modal_phase2.py, run in the workspace whose profile is `second` (see 4_start_phase3.bat):
  1. prepare    : pull tokenizer, the old shard manifest and phase 2's shard hash from HF
  2. build      : CPU container rebuilds the SAME shard set (about 45 min); its hash must equal phase 2's
  2b. fetch_ckpt: waits (polls every 2 min) until phase 2's final checkpoint, step 50,000, is on HF, then downloads it
  3. train      : one H100 loads weights + Adam state, reads batches from step 50,000 of the SAME shuffled order,
                  new LR schedule, checkpoints to HF at runs/gqa_p3/latest, stops at --max-cost
So it can be started while phase 2 is still running: the slow data build overlaps with phase 2's last steps.
Needs a Modal secret `huggingface` (HF_TOKEN with write access) in that workspace.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
REMOTE, DATA = "/repo", "/data"
HF_REPO = "bekhruzsuleyman/alethic-300m"
SRC_DIR = "runs/gqa_p2/latest"                        # the phase-2 final checkpoint on HF
RUN = "gqa_p3"
P3 = Path(DATA) / "_p3"                                # stage markers, plan and the old manifest live here
P2_FINAL_STEP = 50_000                                 # phase 2 must have finished all its steps

app = modal.App("alethic-phase3")
volume = modal.Volume.from_name("alethic-data", create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

_ENV = {"ALETHIC_DATA_DIR": DATA, "PYTHONPATH": f"{REMOTE}/src", "PYTHONUNBUFFERED": "1"}
_IGNORE = [".venv", ".git", "data", "**/__pycache__", "**/.pytest_cache", "*.egg-info"]
_REQ = REPO / "requirements.txt"
_PKGS = [ln.strip() for ln in _REQ.read_text().splitlines()
         if ln.strip() and not ln.startswith("#") and not ln.startswith("torch")] if _REQ.exists() else []
image = (modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(str(_REQ)).env(_ENV)
         .workdir(REMOTE).add_local_dir(str(REPO), REMOTE, ignore=_IGNORE)) if _REQ.exists() else None
cpu_image = (modal.Image.debian_slim(python_version="3.12").pip_install(*_PKGS).env(_ENV)
             .workdir(REMOTE).add_local_dir(str(REPO), REMOTE, ignore=_IGNORE)) if _REQ.exists() else None

# Phase 3 plan: steps = what is left of one epoch after the phase-2 offset (computed in `build`, about 30,000 steps,
# 1.97B tokens, about 3.7 h on an H100). LR: re-warm to 5e-5, cosine to 5e-6 (min_lr = peak / 10 in cli.py).
WARMUP, PEAK_LR = 200, 5e-5
SEQ_LEN, BATCH_SEQS = 4096, 16                         # must equal configs/train_p2.yaml


@app.function(image=image, cpu=4, memory=8192, timeout=1800)
def selftest() -> tuple[int, str]:
    """Stage 0 (about $0.02): run the training, loader and config tests on Linux; a failure stops the pipeline."""
    import subprocess as sp, sys
    proc = sp.run([sys.executable, "-m", "pytest", "-q", "--basetemp", "/tmp/pt", "tests/test_train.py",
                   "tests/test_loader.py", "tests/test_config.py", "tests/test_model.py"],
                  cwd=REMOTE, capture_output=True, text=True)
    return proc.returncode, "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])


@app.function(image=cpu_image, cpu=2, memory=8192, timeout=3 * 3600, volumes={DATA: volume}, secrets=[hf_secret])
def prepare() -> dict:
    """Stage 1: download the tokenizer, the old shard manifest and phase 2's shard hash from HF into the Volume."""
    from huggingface_hub import hf_hub_download, snapshot_download
    volume.reload()
    if (P3 / "prepared").exists():
        return {"skipped": True}
    P3.mkdir(parents=True, exist_ok=True)
    snapshot_download(HF_REPO, allow_patterns=["tokenizer/*", "tokenizer/**"], local_dir=DATA)
    hf_hub_download(HF_REPO, f"{SRC_DIR}/manifest.json", local_dir=DATA, force_download=True)   # only for the shard hash
    old = hf_hub_download(HF_REPO, "shards/manifest.json", local_dir=str(P3 / "hf"))
    shutil.copyfile(old, P3 / "seen_manifest.json")
    need = [Path(DATA) / "tokenizer/specials.json", Path(DATA) / "tokenizer/alethic-tokenizer-v1/FROZEN.json",
            Path(DATA) / SRC_DIR / "manifest.json"]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        raise RuntimeError(f"HF repo is missing files the run needs: {missing}")
    man = json.loads((Path(DATA) / SRC_DIR / "manifest.json").read_text())
    # the shard set never changes during phase 2, so any phase-2 checkpoint gives the right hash; the offset is fixed
    (P3 / "plan.json").write_text(json.dumps({"offset": P2_FINAL_STEP, "expect_shards_sha256": man["shards_sha256"]}))
    (P3 / "prepared").write_text("ok")
    volume.commit()
    return {"phase2_step_seen": int(man["step"]), "shards_sha256": man["shards_sha256"][:12]}


@app.function(image=cpu_image, cpu=16, memory=32768, timeout=6 * 3600, volumes={DATA: volume}, secrets=[hf_secret],
              retries=modal.Retries(max_retries=2, initial_delay=10.0, backoff_coefficient=1.0))
def build() -> dict:
    """Stage 2: build all shards on CPU, then require that the old (seen) shards are found in the new set."""
    from alethic import cli
    volume.reload()
    if (P3 / "built").exists():
        return json.loads((P3 / "built").read_text())
    code = cli.main(["data-build", "--data", f"{REMOTE}/configs/data.yaml", "--target-scale", "1.0", "--workers", "16"],
                    on_checkpoint=lambda _t: volume.commit())
    volume.commit()
    if code != 0:
        raise RuntimeError(f"data-build exit code {code}")
    from alethic.data.loader import seen_in_new_manifest
    excluded, info = seen_in_new_manifest(P3 / "seen_manifest.json", Path(DATA) / "shards/manifest.json")
    info["excluded_share"] = info["excluded_tokens"] / info["old_train_tokens"]
    print(json.dumps(info))
    # the seen row groups must be found (>= 95% of old tokens) and not exceed 3x (a wrong match would waste data)
    if not 0.95 <= info["excluded_share"] <= 3.0 or info["unseen_train_tokens"] < 5_000 * SEQ_LEN * BATCH_SEQS:
        raise RuntimeError(f"seen-data check failed: {info}")
    (P3 / "exclude.json").write_text(json.dumps({"exclude": sorted(excluded)}))
    # Phase 3 must read the same shuffled order as phase 2: same shard list and sizes => same hash and same permutation.
    from alethic.config import config_hash
    from alethic.data.loader import train_shards_from_manifest
    plan = json.loads((P3 / "plan.json").read_text())
    shards, _ = train_shards_from_manifest(Path(DATA) / "shards", excluded)
    got = config_hash({"shards": [(f"{p.parent.name}/{p.name}", p.stat().st_size) for p in shards]})
    if got != plan["expect_shards_sha256"]:
        raise RuntimeError("the rebuilt shard set is not identical to phase 2's (hash differs), so the data order "
                           f"cannot continue. Nothing was spent on GPU. got {got}, phase 2 had {plan['expect_shards_sha256']}")
    total_tokens = sum(p.stat().st_size for p in shards) // 2
    per_epoch = ((total_tokens - 1) // SEQ_LEN) // BATCH_SEQS            # steps in one pass over all windows
    steps = (per_epoch - plan["offset"] - 50) // 100 * 100               # stay inside the epoch: nothing repeats
    if steps < 5_000:
        raise RuntimeError(f"only {steps} unseen steps are left in this epoch (epoch {per_epoch}, offset {plan['offset']})")
    plan["steps"] = steps
    (P3 / "plan.json").write_text(json.dumps(plan))
    info["steps_planned"] = steps
    (P3 / "built").write_text(json.dumps(info))
    volume.commit()
    return info


@app.function(image=cpu_image, cpu=1, memory=4096, timeout=5 * 3600, volumes={DATA: volume}, secrets=[hf_secret])
def fetch_ckpt() -> dict:
    """Stage 2b: wait until phase 2's FINAL checkpoint (step 50,000) is on HF, then download it. Polls every 2 min."""
    from huggingface_hub import hf_hub_download, snapshot_download
    volume.reload()
    if (P3 / "ckpt_ready").exists():
        return {"skipped": True}
    deadline = time.time() + 4.5 * 3600
    while time.time() < deadline:
        m = json.loads(Path(hf_hub_download(HF_REPO, f"{SRC_DIR}/manifest.json", local_dir=str(P3 / "peek"),
                                            force_download=True)).read_text())
        if int(m["step"]) == P2_FINAL_STEP:
            snapshot_download(HF_REPO, allow_patterns=[f"{SRC_DIR}/*"], local_dir=DATA, force_download=True)
            got = json.loads((Path(DATA) / SRC_DIR / "manifest.json").read_text())
            if int(got["step"]) == P2_FINAL_STEP and (Path(DATA) / SRC_DIR / "state.pt").is_file():
                (P3 / "ckpt_ready").write_text("ok")
                volume.commit()
                return {"step": int(got["step"]), "checkpoint_gb": round((Path(DATA) / SRC_DIR / "state.pt").stat().st_size / 1e9, 2)}
        print(f"phase 2 is at step {m['step']}; waiting for {P2_FINAL_STEP}", flush=True)
        time.sleep(120)
    raise RuntimeError(f"phase 2's final checkpoint (step {P2_FINAL_STEP}) did not appear on HF within 4.5 h")


def _push(folder: Path | None, run_dir: Path) -> None:
    """Upload logs (and the checkpoint folder, if given) to HF; failures are printed, never raised."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        logs = [p for p in ("train_log.jsonl", "val_log.jsonl", "runs.jsonl", "status.json") if (run_dir / p).is_file()]
        for name in logs:
            api.upload_file(path_or_fileobj=str(run_dir / name), path_in_repo=f"runs/{RUN}/{name}", repo_id=HF_REPO,
                            repo_type="model", commit_message=f"{RUN}: {name}")
        if folder is not None:
            api.upload_folder(folder_path=str(folder), path_in_repo=f"runs/{RUN}/latest", repo_id=HF_REPO,
                              repo_type="model", delete_patterns=["*"], commit_message=f"{RUN}: {folder.name}")
            print(f"[hf] pushed {folder.name}", flush=True)
    except Exception as err:
        print(f"[hf] push warning: {err}", flush=True)


def _last_row(run_dir: Path) -> dict | None:
    """The newest train_log.jsonl row, or None."""
    f = run_dir / "train_log.jsonl"
    if not f.is_file():
        return None
    lines = f.read_text().strip().splitlines()
    return json.loads(lines[-1]) if lines else None


@app.function(image=image, gpu="H100", cpu=2, memory=12288, timeout=24 * 3600, volumes={DATA: volume},
              secrets=[hf_secret], retries=modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=1.0))
def train(max_cost: float = 20.0) -> int:
    """Stage 3: phase-3 training. Always --resume auto, so a retry continues from this run's own checkpoint."""
    from alethic import cli
    volume.reload()
    run_dir = Path(DATA) / "runs" / RUN
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {"bad": False}

    def on_checkpoint(folder: Path) -> None:
        """After each checkpoint: commit to the Volume, push to HF unless the loss looks broken."""
        volume.commit()
        row = _last_row(run_dir)
        if row is None or not math.isfinite(row["loss"]):
            print("[p3] loss not finite: NOT pushing this checkpoint", flush=True)
            return
        _push(folder, run_dir)

    def heartbeat() -> None:
        """Every 10 min: push small logs; stop the run (with a save) if the loss is NaN/inf."""
        while True:
            time.sleep(600)
            row = _last_row(run_dir)
            if row is None:
                continue
            (run_dir / "status.json").write_text(json.dumps({**row, "unix_time": time.time()}))
            if not math.isfinite(row["loss"]) and not state["bad"]:
                state["bad"] = True
                print("[p3] NaN/inf loss: sending SIGTERM", flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
            _push(None, run_dir)

    threading.Thread(target=heartbeat, daemon=True).start()
    plan = json.loads((P3 / "plan.json").read_text())
    args = ["train", "--model", f"{REMOTE}/configs/model_gqa.yaml", "--train", f"{REMOTE}/configs/train_p2.yaml",
            "--run-name", RUN, "--resume", "auto", "--init-from", f"{DATA}/{SRC_DIR}",
            "--exclude-manifest", str(P3 / "exclude.json"), "--steps", str(plan["steps"]),
            "--warmup-steps", str(WARMUP), "--peak-lr", str(PEAK_LR), "--max-cost", str(max_cost),
            "--data-offset-steps", str(plan["offset"]), "--expect-shards-sha256", plan["expect_shards_sha256"]]
    try:
        return cli.main(args, on_checkpoint=on_checkpoint)
    finally:
        volume.commit()
        last = None
        ck = Path(DATA) / "runs" / RUN / "checkpoints"
        if ck.is_dir():
            done = sorted(p for p in ck.iterdir() if p.is_dir() and p.name.startswith("step_") and (p / "manifest.json").is_file())
            last = done[-1] if done else None
        row = _last_row(run_dir)
        if last is not None and row is not None and math.isfinite(row["loss"]):
            _push(last, run_dir)         # final checkpoint, e.g. after the cost cap stops the run


@app.function(image=cpu_image, cpu=1, memory=1024, timeout=24 * 3600, secrets=[hf_secret])
def pipeline(max_cost: float = 20.0) -> dict:
    """All stages in order; each skips itself when its marker exists on the Volume."""
    code, tail = selftest.remote()
    print(tail, flush=True)
    if code != 0:
        raise RuntimeError("selftest failed; nothing was spent on GPU or data. See the log above.")
    out = {"prepare": prepare.remote(), "build": build.remote(), "ckpt": fetch_ckpt.remote()}
    print(out, flush=True)
    out["train_exit_code"] = train.remote(max_cost)
    return out


@app.local_entrypoint()
def start(max_cost: float = 20.0) -> None:
    """modal run --detach deploy/modal_phase3.py::start"""
    if max_cost <= 0 or max_cost > 29:
        raise SystemExit("--max-cost must be between 0 and 29 dollars")
    call = pipeline.spawn(max_cost)
    print("spawned pipeline:", call.object_id)
