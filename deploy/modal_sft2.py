"""SFT-2 (5k set) for Alethic-300M on Modal: mix the checked sources into one 5k set, then fine-tune the base model.

Start (double-click deploy\18_train_sft2_5k_second_account.bat):  modal run --detach deploy/modal_sft2.py::start
Stages (one detached call, closing the window is safe):
  1. selftest : the SFT unit tests on Linux (CPU, about $0.02). A failure stops everything.
  2. assemble : CPU. Reads T/E, B, D from HF and C, P from the Volume (the DeepSeek runs), and F, G, identity and the hand
                data from the image; tools/expand/compose.py mixes them to about 5,000 samples -> Volume /data/sft2_5k/.
  3. train    : one H100: base checkpoint + tokenizer from HF, 3 epochs, keep the best, push runs/sft2_5k to HF.
Needs the Modal secret `huggingface` in the workspace you run it in. Hard stop: 40-minute training timeout (about $2.8).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
REMOTE, DATA, SFT_DATA, MIX = "/repo", "/data", "/sft_data", "/data/sft2_5k"
HF_REPO = "bekhruzsuleyman/alethic-300m"
TOKENIZER = "tokenizer/alethic-tokenizer-v1/alethic-tokenizer-v1.model"
HF_FILES = {"T": "sft_expand_natural_rechecked/built_T.jsonl", "E": "sft_expand_natural_rechecked/built_E.jsonl",
            "B": "sft_code_full_rechecked/built_B.jsonl", "D": "sft_traces_pilot_rechecked/built_D.jsonl"}
VOLUME_FILES = {"C": "expand_think_full/built_C.jsonl", "P": "expand_proj_full/built_P.jsonl"}

app = modal.App("alethic-sft2")
volume = modal.Volume.from_name("alethic-data", create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

_ENV = {"ALETHIC_DATA_DIR": DATA, "PYTHONPATH": f"{REMOTE}/src:{REMOTE}", "PYTHONUNBUFFERED": "1"}
_IGNORE = [".venv", ".git", "data", "docs", "**/__pycache__", "**/.pytest_cache", "*.egg-info"]
_DATA_IGNORE = ["*_preview.txt", "*_report.json", "token_budget_real.json", "plain_batch1_*", "plain_batch2_*"]
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
]


def _read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


@app.function(image=image, cpu=4, memory=8192, timeout=1800)
def selftest() -> tuple[int, str]:
    import subprocess as sp
    import sys
    proc = sp.run([sys.executable, "-m", "pytest", "-q", "--basetemp", "/tmp/pt", "tests/test_sft_data.py", "tests/test_sft_train.py",
                   "tools/expand/tests/test_compose.py", "tools/expand/tests/test_mapgen.py"], cwd=REMOTE, capture_output=True, text=True)
    return proc.returncode, "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])


@app.function(image=image, cpu=2, memory=4096, timeout=1800, volumes={DATA: volume}, secrets=[hf_secret])
def assemble(total: int = 5000) -> dict:
    from huggingface_hub import hf_hub_download
    from tools.expand import compose

    volume.reload()
    src = {}
    for name, rel in HF_FILES.items():
        src[name] = _read(hf_hub_download(HF_REPO, rel, repo_type="model", local_dir="/tmp/hf"))
    for name, rel in VOLUME_FILES.items():
        p = Path(DATA) / rel
        if not p.is_file():
            raise RuntimeError(f"{p} is missing: the DeepSeek run for {name} has not written its data to the Volume")
        src[name] = _read(p)
    built = Path(SFT_DATA)
    src["F"], src["G"] = _read(built / "maps_F.jsonl"), _read(built / "maps_G.jsonl")
    src["identity"] = _read(built / "identity_en_uz_built.jsonl")
    src["hand"] = [s for f in sorted(built.glob("*.jsonl")) if not f.name.startswith(("maps_", "identity_")) for s in _read(f)]
    print({k: len(v) for k, v in src.items()}, flush=True)
    out = compose.compose(src, total=total, seed=7)
    if Path(MIX).exists():
        shutil.rmtree(MIX)
    Path(MIX).mkdir(parents=True)
    for name, samples in out.items():
        (Path(MIX) / f"{name}.jsonl").write_text("".join(json.dumps(s, ensure_ascii=True) + "\n" for s in samples), encoding="utf-8")
    volume.commit()
    rep = compose.report(out)
    print("SFT-2 set:", json.dumps(rep), flush=True)
    return rep


@app.function(image=image, gpu="H100", cpu=4, memory=32768, timeout=2400, volumes={DATA: volume}, secrets=[hf_secret])
def train(run: str, base: str, epochs: int, peak_lr: float, batch_size: int) -> dict:
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
    files = sorted(str(p) for p in Path(MIX).glob("*.jsonl"))
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
    try:
        HfApi().upload_folder(folder_path=str(out), path_in_repo=f"runs/{run}", repo_id=HF_REPO, repo_type="model",
                              commit_message=f"{run}: best epoch {report['best_epoch']}, val {report['best_val_loss']:.4f}")
        HfApi().upload_folder(folder_path=MIX, path_in_repo=f"sft2_5k_data", repo_id=HF_REPO, repo_type="model",
                              commit_message=f"{run}: the mixed 5k set")
        print(f"[hf] pushed runs/{run} and sft2_5k_data", flush=True)
    except Exception as err:
        print(f"[hf] push warning: {err}", flush=True)
    summary = {k: report[k] for k in ("best_val_loss", "best_epoch", "epochs_run", "samples")}
    print(json.dumps(summary, indent=1), flush=True)
    return summary


@app.function(image=image, cpu=1, memory=1024, timeout=3600 + 1800, secrets=[hf_secret])
def pipeline(run: str, base: str, epochs: int, peak_lr: float, batch_size: int, total: int) -> dict:
    code, tail = selftest.remote()
    print(tail, flush=True)
    if code != 0:
        raise RuntimeError("selftest failed; nothing was spent on GPU. See the log above.")
    print("assembled:", assemble.remote(total), flush=True)
    return train.remote(run, base, epochs, peak_lr, batch_size)


@app.local_entrypoint()
def start(run: str = "sft2_5k", base: str = "runs/gqa_p3/latest", epochs: int = 3, peak_lr: float = 3e-5,
          batch_size: int = 8, total: int = 5000) -> None:
    """modal run --detach deploy/modal_sft2.py::start"""
    call = pipeline.spawn(run, base, epochs, peak_lr, batch_size, total)
    print("spawned pipeline:", call.object_id)
