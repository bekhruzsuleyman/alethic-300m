"""SFT-3 for Alethic-300M on Modal: stage A (100k open chat) then stage B (GOLD x3, error traces, identity, chat replay).

Start (double-click deploy\24_train_sft3_second_account.bat):  modal run --detach deploy/modal_sft3.py::start
One detached call, closing the window is safe:
  1. selftest : the SFT unit tests on Linux (CPU, cents). A failure stops everything before any GPU money is spent.
  2. assemble : CPU. Builds the stage B set from the GOLD files in the image and 2,000 chat examples from stage A -> Volume /data/sft3/stage_b.
  3. stage A  : one H100, the base checkpoint, 1 epoch over the first a_n (default 100,000, all of it) lines of /data/sft3/stage_a_chat.jsonl (uploaded by the .bat). Saved to the Volume and HF.
  4. stage B  : one H100, starts from stage A's best weights, 2 epochs, keeps the best epoch. Saved to the Volume and HF.
Needs the Modal secret `huggingface`. Hard stops: stage A 3 hours, stage B 1 hour (about $12 and $4 at most).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
REMOTE, DATA, SFT_DATA = "/repo", "/data", "/sft_data"
A_FILE, A_USED, B_DIR = "/data/sft3/stage_a_chat.jsonl", "/data/sft3/stage_a_used.jsonl", "/data/sft3/stage_b"
B2_DIR = "/data/sft3/stage_b2"   # stage B again, with the program traces (start_b)
HF_REPO = "bekhruzsuleyman/alethic-300m"
TOKENIZER = "tokenizer/alethic-tokenizer-v1/alethic-tokenizer-v1.model"

app = modal.App("alethic-sft3")
volume = modal.Volume.from_name("alethic-data", create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

_ENV = {"ALETHIC_DATA_DIR": DATA, "PYTHONPATH": f"{REMOTE}/src:{REMOTE}", "PYTHONUNBUFFERED": "1"}
_IGNORE = [".venv", ".git", "data", "docs", "reports", "dist", "build", "work", "_dev_tmp", "**/__pycache__", "**/.pytest_cache", "*.egg-info"]
_DATA_IGNORE = ["*_preview.txt", "*_report.json", "token_budget_real.json"]
image = (modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(str(REPO / "requirements.txt")).env(_ENV)
         .workdir(REMOTE).add_local_dir(str(REPO), REMOTE, ignore=_IGNORE)
         .add_local_dir(str(REPO / "docs/sft/record/built"), SFT_DATA, ignore=_DATA_IGNORE))

PROMPTS = [
    "What is the capital of Japan?",
    "O'zbekistonning poytaxti qayer?",
    "Write a Python function that returns the square of a number.",
    "What will the weather be in Tashkent tomorrow?",
    "Explain in one sentence what a hash table is.",
    "Who are you and who made you?",
    "Sen kimsan va seni kim yaratgan?",
]


@app.function(image=image, cpu=4, memory=8192, timeout=1800)
def selftest() -> tuple[int, str]:
    import subprocess as sp
    import sys
    proc = sp.run([sys.executable, "-m", "pytest", "-q", "--basetemp", "/tmp/pt", "tests/test_sft_data.py", "tests/test_sft_train.py",
                   "tests/test_sft_chain.py", "tests/test_sft3_mix.py"], cwd=REMOTE, capture_output=True, text=True)
    return proc.returncode, "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])


@app.function(image=image, cpu=2, memory=8192, timeout=1800, volumes={DATA: volume})
def assemble(a_n: int = 100000, replay_n: int = 2000, gold_repeat: int = 3) -> dict:
    from tools.sft3 import mix

    volume.reload()
    if not Path(A_FILE).is_file():
        raise RuntimeError(f"{A_FILE} is missing: the stage A chat file was not uploaded to the Volume")
    print("stage A examples used:", mix.take_first(A_FILE, A_USED, a_n), flush=True)
    built = mix.build_stage_b(SFT_DATA, A_USED, gold_repeat=gold_repeat, replay_n=replay_n, seed=7)
    if Path(B_DIR).exists():
        shutil.rmtree(B_DIR)
    mix.write_mix(built, B_DIR)
    volume.commit()
    rep = mix.report(built)
    print("stage B set:", json.dumps(rep), flush=True)
    return rep


@app.function(image=image, cpu=2, memory=8192, timeout=1800, volumes={DATA: volume})
def assemble_b(replay_n: int = 2000, gold_repeat: int = 2) -> dict:
    """Stage B only: the chat replay comes from the stage A file already on the Volume, so nothing is uploaded again."""
    from tools.sft3 import mix

    volume.reload()
    if not Path(A_USED).is_file():
        raise RuntimeError(f"{A_USED} is missing: run the full pipeline (start) once first")
    built = mix.build_stage_b(SFT_DATA, A_USED, gold_repeat=gold_repeat, replay_n=replay_n, seed=7)
    if Path(B2_DIR).exists():
        shutil.rmtree(B2_DIR)
    mix.write_mix(built, B2_DIR)
    volume.commit()
    rep = mix.report(built)
    print("stage B set:", json.dumps(rep), flush=True)
    return rep


def _train(run: str, init: str, files: list[str], epochs: int, peak_lr: float, batch_size: int, val_fraction: float, base_repo_dir: str | None) -> dict:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from alethic.config import sha256_file
    from alethic.sft import train as sft

    volume.reload()
    if base_repo_dir:
        snapshot_download(HF_REPO, allow_patterns=[f"{base_repo_dir}/*"], local_dir=DATA)
    hf_hub_download(HF_REPO, TOKENIZER, local_dir=DATA)
    tok = Path(DATA) / TOKENIZER
    print(f"{run}: init {init}, tokenizer sha {sha256_file(tok)[:12]}, {len(files)} data files", flush=True)
    out = Path(DATA) / "runs" / run
    if out.exists():
        shutil.rmtree(out)
    cfg = sft.parse_args(["--init-from", init, "--model", f"{REMOTE}/configs/model_gqa.yaml", "--tokenizer", str(tok), "--data", *files,
                          "--out", str(out), "--epochs", str(epochs), "--peak-lr", str(peak_lr), "--batch-size", str(batch_size),
                          "--val-fraction", str(val_fraction), "--max-micro-tokens", "8192", "--device", "cuda",
                          *sum((["--sample-prompt", p] for p in PROMPTS), [])])
    report = sft.run_sft(cfg)
    volume.commit()
    try:
        HfApi().upload_folder(folder_path=str(out), path_in_repo=f"runs/{run}", repo_id=HF_REPO, repo_type="model",
                              commit_message=f"{run}: best epoch {report['best_epoch']}, val {report['best_val_loss']:.4f}")
        print(f"[hf] pushed runs/{run}", flush=True)
    except Exception as err:
        print(f"[hf] push warning: {err}", flush=True)
    summary = {k: report[k] for k in ("best_val_loss", "best_epoch", "epochs_run", "steps")}
    print(json.dumps(summary, indent=1), flush=True)
    return summary


@app.function(image=image, gpu="H100", cpu=4, memory=49152, timeout=3 * 3600, volumes={DATA: volume}, secrets=[hf_secret])
def train_a(run: str, base: str, peak_lr: float, batch_size: int) -> dict:
    return _train(run, f"{DATA}/{base}", [A_USED], 1, peak_lr, batch_size, 0.01, base)


@app.function(image=image, gpu="H100", cpu=4, memory=32768, timeout=3600, volumes={DATA: volume}, secrets=[hf_secret])
def train_b(run: str, init_run: str, epochs: int, peak_lr: float, batch_size: int, b_dir: str = B_DIR) -> dict:
    volume.reload()
    files = sorted(str(p) for p in Path(b_dir).glob("*.jsonl"))
    return _train(run, f"{DATA}/runs/{init_run}", files, epochs, peak_lr, batch_size, 0.1, None)


@app.function(image=image, cpu=1, memory=1024, timeout=6 * 3600, secrets=[hf_secret])
def pipeline(base: str, a_n: int, a_lr: float, a_batch: int, b_epochs: int, b_lr: float, b_batch: int) -> dict:
    code, tail = selftest.remote()
    print(tail, flush=True)
    if code != 0:
        raise RuntimeError("selftest failed; nothing was spent on GPU. See the log above.")
    print("assembled:", assemble.remote(a_n), flush=True)
    a = train_a.remote("sft3a", base, a_lr, a_batch)
    print("stage A done:", a, flush=True)
    b = train_b.remote("sft3b", "sft3a", b_epochs, b_lr, b_batch)
    print("stage B done:", b, flush=True)
    return {"a": a, "b": b}


@app.function(image=image, cpu=1, memory=1024, timeout=3 * 3600, secrets=[hf_secret])
def pipeline_b(init_run: str, run: str, epochs: int, lr: float, batch: int, gold_repeat: int) -> dict:
    code, tail = selftest.remote()
    print(tail, flush=True)
    if code != 0:
        raise RuntimeError("selftest failed; nothing was spent on GPU. See the log above.")
    print("assembled:", assemble_b.remote(2000, gold_repeat), flush=True)
    b = train_b.remote(run, init_run, epochs, lr, batch, B2_DIR)
    print("stage B done:", b, flush=True)
    return {"b": b}


@app.local_entrypoint()
def start_b(init_run: str = "sft3a", run: str = "sft3c", b_epochs: int = 2, b_lr: float = 3e-5, b_batch: int = 8, gold_repeat: int = 2) -> None:
    """Stage B again from the finished stage A, with the program traces: modal run --detach deploy/modal_sft3.py::start_b"""
    call = pipeline_b.spawn(init_run, run, b_epochs, b_lr, b_batch, gold_repeat)
    print("spawned stage B pipeline:", call.object_id)


@app.local_entrypoint()
def start(base: str = "runs/gqa_p3/latest", a_n: int = 100000, a_lr: float = 5e-5, a_batch: int = 16, b_epochs: int = 2, b_lr: float = 3e-5, b_batch: int = 8) -> None:
    """modal run --detach deploy/modal_sft3.py::start"""
    call = pipeline.spawn(base, a_n, a_lr, a_batch, b_epochs, b_lr, b_batch)
    print("spawned pipeline:", call.object_id)
