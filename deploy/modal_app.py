"""Modal launcher for Alethic-300M (used while the $30/month free Starter credits last).

This file is the only place that imports `modal`; the alethic package itself stays plain Python,
so the same CLI later runs unchanged on Vast.ai. All large files live on one private Modal Volume,
`alethic-data`, mounted at /data (= ALETHIC_DATA_DIR).

Run from the repo root (each call spends credits, so the owner approves first):
    modal run deploy/modal_app.py::tests                       # all tests on Linux CPU, about $0.01
    modal run deploy/modal_app.py::probe                       # checks Volume rename/fsync/commit, < $0.01
    modal run deploy/modal_app.py::train --model sgca --max-cost 5 --stop-step 2000
    modal run deploy/modal_app.py::tokenizer_sample --streams fineweb_edu,math,uzbek_latn,uzbek_cyrl
    modal run deploy/modal_app.py::tokenizer_train

Prices used for the cost guard (modal.com/pricing, 2026-11-09):
    H100 $0.001097/s = $3.949/h; CPU $0.0000131/core/s = $0.0472/h per physical core (2 vCPU);
    memory $0.00000222/GiB/s = $0.0080/h per GiB.
    Train container = H100 + 4 cores + 32 GiB = 3.949 + 0.189 + 0.256 = $4.39/h.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
REMOTE = "/repo"
DATA = "/data"

app = modal.App("alethic-300m")
volume = modal.Volume.from_name("alethic-data", create_if_missing=True, version=2)
# Created by the owner in the Modal dashboard; holds HF_TOKEN. Never print its value.
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

# Same pinned versions as the laptop; on Linux the torch wheel from PyPI is the CUDA build.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
    .env({"ALETHIC_DATA_DIR": DATA, "PYTHONPATH": f"{REMOTE}/src", "PYTHONUNBUFFERED": "1"})
    .workdir(REMOTE)
    .add_local_dir(str(REPO), REMOTE, ignore=[".venv", ".git", "data", "**/__pycache__", "**/.pytest_cache", "*.egg-info"])
)

# Data and tokenizer work needs no torch: a smaller image that builds and starts faster.
# This file is imported again inside each container (as /root/modal_app.py), where the repo's
# requirements.txt is not at REPO; the image is only built from the laptop, so read it only there.
_REQ = REPO / "requirements.txt"
CPU_PACKAGES = [line.strip() for line in _REQ.read_text().splitlines()
                if line.strip() and not line.startswith("#") and not line.startswith("torch")] if _REQ.exists() else []
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*CPU_PACKAGES)
    .env({"ALETHIC_DATA_DIR": DATA, "PYTHONPATH": f"{REMOTE}/src", "PYTHONUNBUFFERED": "1"})
    .workdir(REMOTE)
    .add_local_dir(str(REPO), REMOTE, ignore=[".venv", ".git", "data", "**/__pycache__", "**/.pytest_cache", "*.egg-info"])
)


@app.function(image=image, cpu=4, memory=8192, timeout=1800)
def run_tests() -> tuple[int, str]:
    """Run the full test suite inside the Modal image; returns (exit code, last lines of output)."""
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--basetemp", "/tmp/pt", "tests"],
                          cwd=REMOTE, capture_output=True, text=True)
    return proc.returncode, "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])


@app.function(image=image, cpu=0.5, memory=1024, timeout=600, volumes={DATA: volume}, secrets=[hf_secret])
def probe_write() -> dict:
    """Check what crash safety relies on: fsync and atomic rename of files and folders on the Volume."""
    root = Path(DATA) / "_probe"
    root.mkdir(parents=True, exist_ok=True)
    out = {"hf_token_present": bool(os.environ.get("HF_TOKEN"))}   # True/False only, never the value
    tmp, final = root / "a.bin.tmp", root / "a.bin"
    with open(tmp, "wb") as f:
        f.write(os.urandom(1 << 20))
        f.flush()
        os.fsync(f.fileno())
        out["fsync"] = "ok"
    os.replace(tmp, final)
    tmp.write_bytes(b"second")
    os.replace(tmp, final)
    out["file_replace"] = final.read_bytes() == b"second" and not tmp.exists()
    d_tmp, d_final = root / "step_0000001.tmp", root / "step_0000001"
    d_tmp.mkdir(exist_ok=True)
    (d_tmp / "state.pt").write_bytes(b"x" * 1000)
    os.replace(d_tmp, d_final)
    out["dir_replace"] = (d_final / "state.pt").stat().st_size == 1000 and not d_tmp.exists()
    volume.commit()
    return out


@app.function(image=image, cpu=0.5, memory=1024, timeout=600, volumes={DATA: volume})
def probe_read() -> dict:
    """From a fresh container: are the committed files visible? Then remove the probe folder."""
    import shutil
    volume.reload()
    root = Path(DATA) / "_probe"
    out = {"file_visible": (root / "a.bin").exists() and (root / "a.bin").read_bytes() == b"second",
           "dir_visible": (root / "step_0000001" / "state.pt").exists(),
           "tmp_left": [p.name for p in root.glob("*.tmp")]}
    shutil.rmtree(root, ignore_errors=True)
    volume.commit()
    return out


@app.function(image=image, gpu="H100", cpu=2, memory=12288, timeout=24 * 3600, volumes={DATA: volume},
              retries=modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=1.0))
def train_remote(model: str, max_cost: float | None, stop_step: int | None, run_name: str | None,
                 steps: int | None = None, peak_lr: float | None = None, warmup_steps: int | None = None) -> int:
    """Train or resume one model on an H100. Always --resume auto, so a preempted or retried call
    continues from the last checkpoint on the Volume."""
    from alethic import cli
    volume.reload()
    def _commit(folder: Path) -> None:
        print(f"[modal] committing checkpoint {folder.name} to Volume", flush=True)
        volume.commit()
    args = ["train", "--model", f"{REMOTE}/configs/model_{model}.yaml",
            "--train", f"{REMOTE}/configs/train.yaml", "--resume", "auto"]
    for flag, val in [("--max-cost", max_cost), ("--stop-step", stop_step), ("--run-name", run_name),
                      ("--steps", steps), ("--peak-lr", peak_lr), ("--warmup-steps", warmup_steps)]:
        if val is not None:
            args += [flag, str(val)]
    try:
        return cli.main(args, on_checkpoint=_commit)
    finally:
        volume.commit()


@app.function(image=image, gpu="H100", cpu=2, memory=12288, timeout=8 * 3600, volumes={DATA: volume},
              secrets=[hf_secret], retries=modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=1.0))
def m6_pair_remote(max_cost_per_run: float = 10.5) -> dict:
    """M6: 1B tokens (15,259 steps, warmup 763, LR 4e-4) matched pair (sgca then gqa) with intentional
    step-1000 stop/resume test on sgca, 500-step Volume checkpoints, and immediate HF backup after each run."""
    from alethic import cli
    from alethic.train import checkpoint as ckpt
    volume.reload()
    def _commit(folder: Path) -> None:
        print(f"[m6] committing checkpoint {folder.name} to Volume", flush=True)
        volume.commit()
    results = {}
    for model in ("sgca", "gqa"):
        name = f"m6_{model}"
        run_dir = Path(DATA) / "runs" / name
        val_log = run_dir / "val_log.jsonl"
        if val_log.is_file() and any('"step": 15259' in ln for ln in val_log.read_text().splitlines()):
            print(f"[m6] {name} already completed; skipping", flush=True)
            continue
        base_args = ["train", "--model", f"{REMOTE}/configs/model_{model}.yaml",
                     "--train", f"{REMOTE}/configs/train.yaml", "--resume", "auto",
                     "--run-name", name, "--steps", "15259", "--warmup-steps", "763",
                     "--peak-lr", "0.0004", "--max-cost", str(max_cost_per_run)]
        if model == "sgca" and ckpt.latest_checkpoint(run_dir / "checkpoints") is None:
            print("[m6] testing intentional stop at step 1000 before resuming m6_sgca", flush=True)
            cli.main([*base_args, "--stop-step", "1000"], on_checkpoint=_commit)
            volume.commit()
        code = cli.main(base_args, on_checkpoint=_commit)
        volume.commit()
        results[name] = code
        if code == 0:
            try:
                backup_runs_to_hf.local(name)
            except Exception as err:
                print(f"[m6] HF backup warning for {name}: {err}", flush=True)
    return results


@app.function(image=cpu_image, cpu=16, memory=32768, timeout=6 * 3600, volumes={DATA: volume},
              secrets=[hf_secret], retries=modal.Retries(max_retries=2, initial_delay=10.0, backoff_coefficient=1.0))
def data_build_remote(target_scale: float = 0.1667, workers: int = 16) -> int:
    """M2/M3: hash, dedup, decontaminate, and encode shards in parallel with Volume checkpoints."""
    from alethic import cli
    volume.reload()
    args = ["data-build", "--data", f"{REMOTE}/configs/data.yaml",
            "--target-scale", str(target_scale), "--workers", str(workers)]
    try:
        return cli.main(args, on_checkpoint=lambda _t: volume.commit())
    finally:
        volume.commit()


@app.function(image=cpu_image, cpu=2, memory=16384, timeout=4 * 3600, volumes={DATA: volume}, secrets=[hf_secret])
def tok_sample(streams: str) -> int:
    """M1 step 1 on a cheap container: download and build some sample streams."""
    from alethic import cli
    try:
        return cli.main(["tokenizer-build", "--stage", "sample", "--streams", streams])
    finally:
        volume.commit()


@app.function(image=cpu_image, cpu=16, memory=32768, timeout=6 * 3600, volumes={DATA: volume},
              secrets=[hf_secret], retries=modal.Retries(max_retries=2, initial_delay=10.0, backoff_coefficient=1.0))
def tok_train() -> int:
    """M1 steps 2-4: merge the sample, train SentencePiece on 32 threads with EM checkpoints, evaluate, freeze."""
    from alethic.config import load_config
    from alethic.tokenizer import build
    volume.reload()
    try:
        return build.run(load_config(f"{REMOTE}/configs/tokenizer.yaml"), load_config(f"{REMOTE}/configs/data.yaml"),
                         Path(f"{REMOTE}/configs/tokenizer.yaml"),
                         load_config(f"{REMOTE}/configs/train.yaml")["paths"]["specials"],
                         stage="all", threads=32, on_checkpoint=lambda _t: volume.commit())
    finally:
        volume.commit()


@app.local_entrypoint()
def tokenizer_sample(streams: str = "fineweb_edu,math,uzbek_latn,uzbek_cyrl") -> None:
    """modal run deploy/modal_app.py::tokenizer_sample --streams fineweb_edu,math"""
    print("exit code", tok_sample.remote(streams))


@app.local_entrypoint()
def tokenizer_train(spawn: bool = False) -> None:
    """modal run --detach deploy/modal_app.py::tokenizer_train --spawn"""
    print("spawned:", tok_train.spawn().object_id) if spawn else print("exit code", tok_train.remote())


@app.local_entrypoint()
def tests() -> None:
    """modal run deploy/modal_app.py::tests"""
    code, tail = run_tests.remote()
    print(tail, "\nexit code", code)


@app.local_entrypoint()
def probe() -> None:
    """modal run deploy/modal_app.py::probe"""
    print("write:", probe_write.remote(), "\nread:", probe_read.remote())


@app.function(image=cpu_image, cpu=2, memory=4096, timeout=1800, volumes={DATA: volume}, secrets=[hf_secret])
def backup_runs_to_hf(run_names: str = "m4_sgca,m4_gqa") -> dict:
    """Upload shards/manifest.json and the specified runs (logs + latest checkpoint) to private HF repo."""
    from huggingface_hub import HfApi
    volume.reload()
    api = HfApi()
    repo_id = "bekhruzsuleyman/alethic-300m"
    commits = {}
    manifest_path = Path(DATA) / "shards" / "manifest.json"
    if manifest_path.is_file():
        info = api.upload_file(path_or_fileobj=str(manifest_path), path_in_repo="shards/manifest.json",
                               repo_id=repo_id, repo_type="model", commit_message="Upload M3a 1B-token shards/manifest.json")
        commits["shards_manifest"] = info.oid
    for name in [r.strip() for r in run_names.split(",") if r.strip()]:
        run_dir = Path(DATA) / "runs" / name
        if not run_dir.is_dir():
            continue
        ckpt_dir = run_dir / "checkpoints"
        steps = sorted(p.name for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("step_")) if ckpt_dir.is_dir() else []
        latest = steps[-1] if steps else None
        ignore = [f"checkpoints/{s}/**" for s in steps if s != latest] + ["**/*.tmp", "**/*.tmp/**"]
        info = api.upload_folder(folder_path=str(run_dir), path_in_repo=f"runs/{name}", repo_id=repo_id,
                                 repo_type="model", ignore_patterns=ignore,
                                 commit_message=f"Backup {name} logs and latest checkpoint ({latest})")
        commits[name] = info.oid
    return commits


@app.local_entrypoint()
def backup_hf(runs: str = "m4_sgca,m4_gqa") -> None:
    """modal run deploy/modal_app.py::backup_hf --runs m4_sgca,m4_gqa"""
    print(backup_runs_to_hf.remote(runs))


@app.local_entrypoint()
def data_build(target_scale: float = 0.1667, workers: int = 16, spawn: bool = False) -> None:
    """modal run --detach deploy/modal_app.py::data_build --target-scale 0.1667 --spawn"""
    if spawn:
        call = data_build_remote.spawn(target_scale, workers)
        print("spawned data_build_remote:", call.object_id)
    else:
        print("exit code", data_build_remote.remote(target_scale, workers))


@app.local_entrypoint()
def m6_pair(spawn: bool = False) -> None:
    """modal run --detach deploy/modal_app.py::m6_pair --spawn"""
    print("spawned m6_pair_remote:", m6_pair_remote.spawn().object_id) if spawn else print(m6_pair_remote.remote())


@app.local_entrypoint()
def train(model: str = "sgca", max_cost: float = 0.0, stop_step: int = 0, run_name: str = "",
          steps: int = 0, peak_lr: float = 0.0, warmup_steps: int = -1, spawn: bool = False) -> None:
    """modal run --detach deploy/modal_app.py::train --model sgca --max-cost 3 --stop-step 2000 --spawn"""
    if model not in ("sgca", "gqa"):
        raise SystemExit("--model must be sgca or gqa")
    if max_cost <= 0:
        raise SystemExit("set --max-cost (dollars) so a run can never spend more than you allow")
    s_ov, lr_ov, w_ov = (steps or None), (peak_lr or None), (warmup_steps if warmup_steps >= 0 else None)
    if spawn:
        call = train_remote.spawn(model, max_cost, stop_step or None, run_name or None, s_ov, lr_ov, w_ov)
        print("spawned train_remote:", call.object_id)
    else:
        code = train_remote.remote(model, max_cost, stop_step or None, run_name or None, s_ov, lr_ov, w_ov)
        print("exit code", code)
