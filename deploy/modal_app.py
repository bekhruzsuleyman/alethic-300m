"""Modal launcher for Alethic-300M (used while the $30/month free Starter credits last).

This file is the only place that imports `modal`; the alethic package itself stays plain Python,
so the same CLI later runs unchanged on Vast.ai. All large files live on one private Modal Volume,
`alethic-data`, mounted at /data (= ALETHIC_DATA_DIR).

Run from the repo root (each call spends credits, so the owner approves first):
    modal run deploy/modal_app.py::tests                       # all tests on Linux CPU, about $0.01
    modal run deploy/modal_app.py::probe                       # checks Volume rename/fsync/commit, < $0.01
    modal run deploy/modal_app.py::train --model sgca --max-cost 5 --stop-step 2000

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
    .add_local_dir(str(REPO), REMOTE, ignore=[".venv", ".git", "**/__pycache__", "**/.pytest_cache", "*.egg-info"])
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
    os.replace(tmp, final)                      # file rename over nothing
    tmp.write_bytes(b"second")
    os.replace(tmp, final)                      # file rename over an existing file
    out["file_replace"] = final.read_bytes() == b"second" and not tmp.exists()
    d_tmp, d_final = root / "step_0000001.tmp", root / "step_0000001"
    d_tmp.mkdir(exist_ok=True)
    (d_tmp / "state.pt").write_bytes(b"x" * 1000)
    os.replace(d_tmp, d_final)                  # folder rename (checkpoints)
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


@app.function(image=image, gpu="H100", cpu=4, memory=32768, timeout=24 * 3600, volumes={DATA: volume},
              retries=modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=1.0))
def train_remote(model: str, max_cost: float | None, stop_step: int | None, run_name: str | None) -> int:
    """Train or resume one model on an H100. Always --resume auto, so a preempted or retried call
    continues from the last checkpoint on the Volume."""
    from alethic import cli
    volume.reload()
    args = ["train", "--model", f"{REMOTE}/configs/model_{model}.yaml",
            "--train", f"{REMOTE}/configs/train.yaml", "--resume", "auto"]
    if max_cost is not None:
        args += ["--max-cost", str(max_cost)]
    if stop_step is not None:
        args += ["--stop-step", str(stop_step)]
    if run_name:
        args += ["--run-name", run_name]
    try:
        return cli.main(args)
    finally:
        volume.commit()


@app.local_entrypoint()
def tests() -> None:
    """modal run deploy/modal_app.py::tests"""
    code, tail = run_tests.remote()
    print(tail)
    print("exit code", code)


@app.local_entrypoint()
def probe() -> None:
    """modal run deploy/modal_app.py::probe"""
    print("write:", probe_write.remote())
    print("read: ", probe_read.remote())


@app.local_entrypoint()
def train(model: str = "sgca", max_cost: float = 0.0, stop_step: int = 0, run_name: str = "") -> None:
    """modal run deploy/modal_app.py::train --model sgca --max-cost 5 --stop-step 2000"""
    if model not in ("sgca", "gqa"):
        raise SystemExit("--model must be sgca or gqa")
    if max_cost <= 0:
        raise SystemExit("set --max-cost (dollars) so a run can never spend more than you allow")
    code = train_remote.remote(model, max_cost, stop_step or None, run_name or None)
    print("exit code", code)
