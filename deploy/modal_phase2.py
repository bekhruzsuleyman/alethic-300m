"""Phase 2 for Alethic-300M GQA: continue the 1B-token model on new data with Modal credits.

One detached call does everything, in stages that skip themselves when already done:
  1. prepare : pull the tokenizer, the step_0015259 checkpoint and the old shard manifest from HF
  2. build   : CPU container builds the full shard set; checks the old (seen) shards are found in it
  3. train   : one H100 loads weights + Adam state, trains on the unseen shards, checkpoints about
               every 29 min to the Volume and to HF (runs/gqa_p2/latest), stops at --max-cost
Start it with the double-click launcher (2_start_phase2.bat) or:
    modal run --detach deploy/modal_phase2.py::start
Needs a Modal secret named `huggingface` with HF_TOKEN (write access to bekhruzsuleyman/alethic-300m).
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
SRC_CKPT = "runs/m6_gqa/checkpoints/step_0015259"      # the 1B-token GQA checkpoint on HF
RUN = "gqa_p2"
P2 = Path(DATA) / "_p2"                                # stage markers and the old manifest live here

app = modal.App("alethic-phase2")
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

# Training plan (cost math in the chat): 50,000 steps x 65,536 tokens = 3.28B tokens, about 6.1 h on an H100.
STEPS, WARMUP, PEAK_LR = 50_000, 200, 2e-4


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
    """Stage 1: download tokenizer, the 1B checkpoint and the old manifest from HF into the Volume."""
    from huggingface_hub import hf_hub_download, snapshot_download
    volume.reload()
    if (P2 / "prepared").exists():
        return {"skipped": True}
    P2.mkdir(parents=True, exist_ok=True)
    snapshot_download(HF_REPO, allow_patterns=["tokenizer/*", "tokenizer/**", f"{SRC_CKPT}/*"], local_dir=DATA)
    old = hf_hub_download(HF_REPO, "shards/manifest.json", local_dir=str(P2 / "hf"))
    shutil.copyfile(old, P2 / "seen_manifest.json")
    need = [Path(DATA) / "tokenizer/specials.json", Path(DATA) / "tokenizer/alethic-tokenizer-v1/FROZEN.json",
            Path(DATA) / SRC_CKPT / "state.pt", Path(DATA) / SRC_CKPT / "manifest.json"]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        raise RuntimeError(f"HF repo is missing files the run needs: {missing}")
    (P2 / "prepared").write_text("ok")
    volume.commit()
    return {"checkpoint_gb": round((Path(DATA) / SRC_CKPT / "state.pt").stat().st_size / 1e9, 2)}


@app.function(image=cpu_image, cpu=16, memory=32768, timeout=6 * 3600, volumes={DATA: volume}, secrets=[hf_secret],
              retries=modal.Retries(max_retries=2, initial_delay=10.0, backoff_coefficient=1.0))
def build() -> dict:
    """Stage 2: build all shards on CPU, then require that the old (seen) shards are found in the new set."""
    from alethic import cli
    volume.reload()
    if (P2 / "built").exists():
        return json.loads((P2 / "built").read_text())
    code = cli.main(["data-build", "--data", f"{REMOTE}/configs/data.yaml", "--target-scale", "1.0", "--workers", "16"],
                    on_checkpoint=lambda _t: volume.commit())
    volume.commit()
    if code != 0:
        raise RuntimeError(f"data-build exit code {code}")
    from alethic.data.loader import seen_in_new_manifest
    excluded, info = seen_in_new_manifest(P2 / "seen_manifest.json", Path(DATA) / "shards/manifest.json")
    info["excluded_share"] = info["excluded_tokens"] / info["old_train_tokens"]
    print(json.dumps(info))
    # the seen row groups must be found (>= 95% of old tokens) and not exceed 3x (a wrong match would waste data)
    if not 0.95 <= info["excluded_share"] <= 3.0 or info["unseen_train_tokens"] < STEPS * 65_536:
        raise RuntimeError(f"seen-data check failed: {info}")
    (P2 / "exclude.json").write_text(json.dumps({"exclude": sorted(excluded)}))
    (P2 / "built").write_text(json.dumps(info))
    volume.commit()
    return info


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
def train(max_cost: float = 26.5) -> int:
    """Stage 3: phase-2 training. Always --resume auto, so a retry continues from this run's own checkpoint."""
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
            print("[p2] loss not finite: NOT pushing this checkpoint", flush=True)
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
                print("[p2] NaN/inf loss: sending SIGTERM", flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
            _push(None, run_dir)

    threading.Thread(target=heartbeat, daemon=True).start()
    args = ["train", "--model", f"{REMOTE}/configs/model_gqa.yaml", "--train", f"{REMOTE}/configs/train_p2.yaml",
            "--run-name", RUN, "--resume", "auto", "--init-from", f"{DATA}/{SRC_CKPT}",
            "--exclude-manifest", str(P2 / "exclude.json"), "--steps", str(STEPS),
            "--warmup-steps", str(WARMUP), "--peak-lr", str(PEAK_LR), "--max-cost", str(max_cost)]
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
def pipeline(max_cost: float = 26.5) -> dict:
    """All stages in order; each skips itself when its marker exists on the Volume."""
    code, tail = selftest.remote()
    print(tail, flush=True)
    if code != 0:
        raise RuntimeError("selftest failed; nothing was spent on GPU or data. See the log above.")
    out = {"prepare": prepare.remote(), "build": build.remote()}
    print(out, flush=True)
    out["train_exit_code"] = train.remote(max_cost)
    return out


@app.local_entrypoint()
def start(max_cost: float = 26.5) -> None:
    """modal run --detach deploy/modal_phase2.py::start"""
    if max_cost <= 0 or max_cost > 29:
        raise SystemExit("--max-cost must be between 0 and 29 dollars")
    call = pipeline.spawn(max_cost)
    print("spawned pipeline:", call.object_id)
