"""SFT-1 for Alethic-300M on Modal: fine-tune the pretrained model on the built SFT samples.

Start (double-click deploy\6_start_sft1.bat):  modal run --detach deploy/modal_sft1.py::start
Stages (all inside one detached call, so closing the window is safe):
  1. selftest : run the SFT unit tests on Linux (CPU, about $0.02). A failure stops everything.
  2. train    : one H100: download base checkpoint + tokenizer from HF, train, keep the best epoch,
                greedy-sample a few prompts, push report + logs + best weights to HF runs/<run>/.
Needs the Modal secret `huggingface` (HF_TOKEN with write access) in the workspace you run it in.
Cost: image build + about 4 GB download + a few minutes of H100; the hard stop is the 40-minute timeout (about $2.8).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
REMOTE, DATA, SFT_DATA = "/repo", "/data", "/sft_data"
HF_REPO = "bekhruzsuleyman/alethic-300m"
TOKENIZER = "tokenizer/alethic-tokenizer-v1/alethic-tokenizer-v1.model"

app = modal.App("alethic-sft1")
volume = modal.Volume.from_name("alethic-data", create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

_ENV = {"ALETHIC_DATA_DIR": DATA, "PYTHONPATH": f"{REMOTE}/src", "PYTHONUNBUFFERED": "1"}
_IGNORE = [".venv", ".git", "data", "docs", "**/__pycache__", "**/.pytest_cache", "*.egg-info"]
# only the checked, built samples; the older plain_batch1_/plain_batch2_ files are duplicates of plain_batches1-2_cleaned
_DATA_IGNORE = ["*_preview.txt", "*_report.json", "token_budget_real.json", "plain_batch1_*", "plain_batch2_*"]
_REQ = REPO / "requirements.txt"
image = (modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(str(_REQ)).env(_ENV)
         .workdir(REMOTE).add_local_dir(str(REPO), REMOTE, ignore=_IGNORE)
         .add_local_dir(str(REPO / "docs/sft/record/built"), SFT_DATA, ignore=_DATA_IGNORE))

PROMPTS = [
    "What is the capital of Japan?",
    "O'zbekistonning poytaxti qayer?",
    "Write a Python function that returns the square of a number.",
    "What will the weather be in Tashkent tomorrow?",
    "Explain in one sentence what a hash table is.",
]


@app.function(image=image, cpu=4, memory=8192, timeout=1800)
def selftest() -> tuple[int, str]:
    """Stage 1: the SFT tests on Linux; a failure stops the pipeline before any GPU is rented."""
    import subprocess as sp
    import sys
    proc = sp.run([sys.executable, "-m", "pytest", "-q", "--basetemp", "/tmp/pt",
                   "tests/test_sft_data.py", "tests/test_sft_train.py"], cwd=REMOTE, capture_output=True, text=True)
    return proc.returncode, "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])


@app.function(image=image, gpu="H100", cpu=4, memory=32768, timeout=2400, volumes={DATA: volume}, secrets=[hf_secret])
def train(run: str, base: str, epochs: int, peak_lr: float, batch_size: int) -> dict:
    """Stage 2: download base + tokenizer, run SFT, push the results to HF."""
    import os
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from alethic.config import sha256_file
    from alethic.sft import train as sft

    volume.reload()
    snapshot_download(HF_REPO, allow_patterns=[f"{base}/*"], local_dir=DATA)
    hf_hub_download(HF_REPO, TOKENIZER, local_dir=DATA)
    base_dir = Path(DATA) / base
    manifest = json.loads((base_dir / "manifest.json").read_text())
    tok = Path(DATA) / TOKENIZER
    print(f"base {base}: step {manifest['step']}, tokenizer sha {sha256_file(tok)[:12]}", flush=True)

    files = sorted(str(p) for p in Path(SFT_DATA).glob("*.jsonl"))
    print(f"{len(files)} data files: {[Path(f).name for f in files]}", flush=True)
    out = Path(DATA) / "runs" / run
    if out.exists():
        shutil.rmtree(out)
    cfg = sft.parse_args(["--init-from", str(base_dir), "--model", f"{REMOTE}/configs/model_gqa.yaml",
                          "--tokenizer", str(tok), "--data", *files, "--out", str(out),
                          "--epochs", str(epochs), "--peak-lr", str(peak_lr), "--batch-size", str(batch_size),
                          "--max-micro-tokens", "8192", "--device", "cuda",
                          *sum((["--sample-prompt", p] for p in PROMPTS), [])])
    report = sft.run_sft(cfg)
    volume.commit()
    try:                                   # results go to HF so they can be read without any terminal
        api = HfApi()
        api.upload_folder(folder_path=str(out), path_in_repo=f"runs/{run}", repo_id=HF_REPO, repo_type="model",
                          commit_message=f"{run}: best epoch {report['best_epoch']}, val {report['best_val_loss']:.4f}")
        print(f"[hf] pushed runs/{run}", flush=True)
    except Exception as err:               # never lose the Volume copy because the push failed
        print(f"[hf] push warning: {err}", flush=True)
    print(json.dumps({k: report[k] for k in ("best_val_loss", "best_epoch", "epochs_run", "samples")}, indent=1), flush=True)
    return {k: report[k] for k in ("best_val_loss", "best_epoch", "epochs_run")}


@app.function(image=image, cpu=1, memory=1024, timeout=3600, secrets=[hf_secret])
def pipeline(run: str, base: str, epochs: int, peak_lr: float, batch_size: int) -> dict:
    """selftest, then train."""
    code, tail = selftest.remote()
    print(tail, flush=True)
    if code != 0:
        raise RuntimeError("selftest failed; nothing was spent on GPU. See the log above.")
    return train.remote(run, base, epochs, peak_lr, batch_size)


@app.local_entrypoint()
def start(run: str = "sft1", base: str = "runs/gqa_p3/latest", epochs: int = 5, peak_lr: float = 3e-5,
          batch_size: int = 8) -> None:
    """modal run --detach deploy/modal_sft1.py::start"""
    call = pipeline.spawn(run, base, epochs, peak_lr, batch_size)
    print("spawned pipeline:", call.object_id)
