"""Overnight dataset expansion on Modal: DeepSeek API writes T (chats) and E (idea pairs) batches,
tools/expand/pcheck.py checks each one, only passing examples are kept.

Start (double-click deploy/8_expand.bat):  modal run --detach deploy/modal_expand.py::start
Needs two Modal secrets in the workspace you run it in:
  `deepseek`     with DEEPSEEK_API_KEY   (your DeepSeek API key)
  `huggingface`  with HF_TOKEN           (write access; already used by the other deploy scripts)
Spend: stops by itself at --cap dollars (default 6), and stops early if fewer than 25% of examples pass.
Results: Volume `alethic-data` under expand/, and the HF repo folder sft_expand/ (accepted examples, built JSONL, stats).
Rerunning continues where it stopped (batches whose raw reply exists are skipped).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
REMOTE, DATA, PROMPTS = "/repo", "/data", "/prompts"
HF_REPO = "bekhruzsuleyman/alethic-300m"

app = modal.App("alethic-expand")
volume = modal.Volume.from_name("alethic-data", create_if_missing=True, version=2)
ds_secret = modal.Secret.from_name("deepseek", required_keys=["DEEPSEEK_API_KEY"])
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])
_base = (modal.Image.debian_slim(python_version="3.12").apt_install("nodejs").pip_install("requests", "huggingface_hub", "pytest", "sympy"))
_local = lambda img: (img.env({"PYTHONPATH": REMOTE, "PYTHONUNBUFFERED": "1"}).workdir(REMOTE)
                      .add_local_dir(str(REPO / "tools"), REMOTE + "/tools", ignore=["**/__pycache__", "**/.pytest_cache"])
                      .add_local_dir(str(REPO / "docs/sft/record/prompts"), PROMPTS))
image = _local(_base)
# P projects need real compilers: build steps first, local files last.
proj_image = _local(_base.apt_install("gcc", "golang-go", "rustc", "npm").run_commands("npm install -g typescript"))

# USD per 1M tokens (peak prices found 2026-11-17): cache-miss input, cache-hit input, output
RATES = {"flash": (0.30, 0.006, 1.20), "pro": (1.32, 0.044, 3.96)}


@app.function(image=image, cpu=1, memory=1024, timeout=900)
def selftest() -> tuple[int, str]:
    """The expansion and plain-builder tests on Linux; a failure stops the run before any API money is spent."""
    import subprocess as sp
    import sys
    proc = sp.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tools/expand/tests",
                   "tools/sftbuild/tests/test_plain.py", "tools/sftbuild/tests/test_normalize.py"],
                  cwd=REMOTE, capture_output=True, text=True)
    return proc.returncode, "\n".join((proc.stdout + proc.stderr).splitlines()[-12:])


@app.function(image=image, cpu=2, memory=4096, timeout=12 * 3600, volumes={DATA: volume}, secrets=[ds_secret, hf_secret])
def expand(kinds: str, cap: float, workers: int, model: str, limit: int, tag: str = "", seed: int = 7, styles: str = "") -> dict:
    from huggingface_hub import HfApi
    from tools.expand import runner

    key = os.environ["DEEPSEEK_API_KEY"]
    for kv in filter(None, styles.split(",")):                  # e.g. "T=detailed,E=natural"
        k, v = kv.split("=")
        runner.STYLE[k] = v
        runner.BASES[k] = {"T": "deepseek_TE_chats_v2_detailed.md", "E": "deepseek_E_pairs_v2_natural.md"}[k]
    print("styles:", runner.STYLE, "bases:", runner.BASES, flush=True)
    models = runner.list_models(key)
    print("models on this key:", models, flush=True)
    if model == "auto":
        model = next((m for m in models if "flash" in m), models[0])
    tier = "flash" if "flash" in model else "pro"          # unknown names are priced as pro: the safe side
    miss, hit, out = RATES[tier]
    print(f"model {model} (priced as {tier}: {miss}/{hit}/{out} per 1M), cap ${cap}", flush=True)

    per_kind = []
    for kind in kinds.split(","):
        start, total = runner.RANGES[kind]
        per_kind.append(runner.plan_batches(kind, start, total, runner.PER_BATCH[kind], seed=seed))
    batches = [b for group in zip(*per_kind) for b in group] + [b for g in per_kind for b in g[min(map(len, per_kind)):]]
    for b in batches:
        b["base_text"] = (Path(PROMPTS) / b["base"]).read_text(encoding="utf-8")
    if limit:
        batches = batches[:limit]
    print(f"{len(batches)} batches planned", flush=True)

    wd = Path(DATA) / ("expand_" + tag if tag else "expand_v2")
    hf_dir = "sft_expand_" + tag if tag else "sft_expand"
    wd.mkdir(parents=True, exist_ok=True)
    budget = runner.Budget(cap, miss, hit, out)
    prev = wd / "stats.json"
    if prev.is_file():                                      # a rerun keeps counting what was already spent
        budget.spent = json.loads(prev.read_text()).get("spent_usd", 0.0)

    def push() -> None:
        try:
            volume.commit()
            HfApi().upload_folder(folder_path=str(wd), path_in_repo=hf_dir, repo_id=HF_REPO, repo_type="model",
                                  allow_patterns=["accepted_*.txt", "built_*.jsonl", "stats.json", "rejects.jsonl", "raw/*.txt"],
                                  commit_message="sft_expand: progress")
        except Exception as err:
            print("[hf] push warning:", err, flush=True)

    stop = threading.Event()
    def saver() -> None:
        while not stop.wait(900):
            push()
    threading.Thread(target=saver, daemon=True).start()
    try:
        stats = runner.run(batches, lambda p: runner.call_deepseek(p, key, model), wd, budget, workers=workers)
    finally:
        stop.set()
        push()
    print(json.dumps(stats), flush=True)
    return stats


@app.function(image=image, cpu=1, memory=1024, timeout=12 * 3600 + 900, secrets=[ds_secret, hf_secret])
def pipeline(kinds: str, cap: float, workers: int, model: str, limit: int, tag: str = "", seed: int = 7, styles: str = "") -> dict:
    code, tail = selftest.remote()
    print(tail, flush=True)
    if code != 0:
        raise RuntimeError("selftest failed; nothing was spent. See the log above.")
    return expand.remote(kinds, cap, workers, model, limit, tag, seed, styles)


@app.local_entrypoint()
def start(kinds: str = "T,E", cap: float = 6.0, workers: int = 16, model: str = "auto", limit: int = 0) -> None:
    """modal run --detach deploy/modal_expand.py::start"""
    if not 0 < cap <= 40:
        raise SystemExit("--cap must be between 0 and 40 dollars")
    call = pipeline.spawn(kinds, cap, workers, model, limit)
    print("spawned pipeline:", call.object_id)


@app.function(image=image, cpu=1, memory=2048, timeout=1800, volumes={DATA: volume}, secrets=[hf_secret])
def recheck() -> dict:
    """Re-check every saved raw reply of the first run (expand_v2/raw) with the current checker. No DeepSeek call, no cost.
    Writes Volume expand_v3/ and pushes accepted/built/rejects/stats to the HF folder sft_expand_rechecked/."""
    from huggingface_hub import HfApi
    from tools.expand import runner

    volume.reload()
    per_kind = [runner.plan_batches(k, *runner.RANGES[k], runner.PER_BATCH[k], seed=7) for k in ("T", "E")]
    batches = [b for g in per_kind for b in g]
    wd = Path(DATA) / "expand_v3"
    stats = runner.reprocess(batches, Path(DATA) / "expand_v2" / "raw", wd)
    print(json.dumps(stats), flush=True)
    volume.commit()
    HfApi().upload_folder(folder_path=str(wd), path_in_repo="sft_expand_rechecked", repo_id=HF_REPO, repo_type="model",
                          allow_patterns=["accepted_*.txt", "built_*.jsonl", "stats.json", "rejects.jsonl"],
                          commit_message="sft_expand_rechecked: re-parsed with pair-id and pair-dedup fixes")
    return stats


@app.local_entrypoint()
def redo() -> None:
    """modal run deploy/modal_expand.py::redo   (free: re-reads saved replies, calls no API)"""
    print(recheck.remote())


@app.local_entrypoint()
def start_natural(cap: float = 6.0, workers: int = 16) -> None:
    """modal run --detach deploy/modal_expand.py::start_natural
    T = detailed answers (up to 140 words), E = natural wording with no fixed label; new topics (seed 11); folder sft_expand_natural/."""
    call = pipeline.spawn("T,E", cap, workers, "auto", 0, "natural", 11, "T=detailed,E=natural")
    print("spawned pipeline:", call.object_id)


@app.function(image=image, cpu=2, memory=4096, timeout=12 * 3600, volumes={DATA: volume}, secrets=[ds_secret, hf_secret])
def traces(n: int, cap: float, workers: int, tag: str = "traces_pilot", start: int = 1, per_batch: int = 5) -> dict:
    """D thinking traces on n generated math problems; only traces with right arithmetic and the right answer are kept."""
    from huggingface_hub import HfApi
    from tools.expand import dtraces, runner

    key = os.environ["DEEPSEEK_API_KEY"]
    models = runner.list_models(key)
    model = next((m for m in models if "pro" in m), models[0])          # D runs on Pro
    miss, hit, out = RATES["flash" if "flash" in model else "pro"]
    probs = dtraces.make_problems(n, start)
    batches = dtraces.plan_batches(probs, per_batch)
    base = (Path(PROMPTS) / "deepseek_D_thinking_traces.md").read_text(encoding="utf-8")
    wd = Path(DATA) / ("expand_" + tag)
    wd.mkdir(parents=True, exist_ok=True)
    budget = runner.Budget(cap, miss, hit, out)
    print(f"model {model}, {len(probs)} problems in {len(batches)} batches, cap ${cap}", flush=True)

    def push() -> None:
        try:
            volume.commit()
            HfApi().upload_folder(folder_path=str(wd), path_in_repo="sft_" + tag, repo_id=HF_REPO, repo_type="model",
                                  allow_patterns=["accepted_*.txt", "built_*.jsonl", "stats.json", "rejects.jsonl", "raw/*.txt"],
                                  commit_message="sft_%s: progress" % tag)
        except Exception as err:
            print("[hf] push warning:", err, flush=True)

    try:
        stats = dtraces.run(batches, lambda p: runner.call_deepseek(p, key, model), wd, budget, workers=workers, base_text=base)
    finally:
        push()
    print(json.dumps(stats), flush=True)
    return stats


@app.function(image=image, cpu=1, memory=1024, timeout=12 * 3600 + 900, secrets=[ds_secret, hf_secret])
def traces_pipeline(n: int, cap: float, workers: int, tag: str, start: int) -> dict:
    code, tail = selftest.remote()
    print(tail, flush=True)
    if code != 0:
        raise RuntimeError("selftest failed; nothing was spent. See the log above.")
    return traces.remote(n, cap, workers, tag, start)


@app.local_entrypoint()
def start_traces_pilot(n: int = 20, cap: float = 1.5, workers: int = 4) -> None:
    """modal run --detach deploy/modal_expand.py::start_traces_pilot   (20 math problems, a few cents; folder sft_traces_pilot/)"""
    call = traces_pipeline.spawn(n, cap, workers, "traces_pilot", 1)
    print("spawned pipeline:", call.object_id)


@app.function(image=image, cpu=1, memory=2048, timeout=1800, volumes={DATA: volume}, secrets=[hf_secret])
def recheck_natural() -> dict:
    """Re-check the saved natural-run replies (expand_natural/raw) with the looser detailed/natural rules. Free."""
    from huggingface_hub import HfApi
    from tools.expand import runner

    volume.reload()
    per_kind = [runner.plan_batches(k, *runner.RANGES[k], runner.PER_BATCH[k], seed=11) for k in ("T", "E")]
    wd = Path(DATA) / "expand_natural_v2"
    stats = runner.reprocess([b for g in per_kind for b in g], Path(DATA) / "expand_natural" / "raw", wd,
                             styles={"T": "detailed", "E": "natural"})
    print(json.dumps(stats), flush=True)
    volume.commit()
    HfApi().upload_folder(folder_path=str(wd), path_in_repo="sft_expand_natural_rechecked", repo_id=HF_REPO, repo_type="model",
                          allow_patterns=["accepted_*.txt", "built_*.jsonl", "stats.json", "rejects.jsonl"],
                          commit_message="natural run re-checked with looser rules")
    return stats


@app.local_entrypoint()
def redo_natural() -> None:
    """modal run deploy/modal_expand.py::redo_natural   (free)"""
    print(recheck_natural.remote())


@app.function(image=image, cpu=2, memory=4096, timeout=12 * 3600, volumes={DATA: volume}, secrets=[ds_secret, hf_secret])
def code(n: int, cap: float, workers: int, tag: str = "code_pilot", start: int = 1, per_batch: int = 10, seed: int = 7) -> dict:
    """B code examples on DeepSeek Pro; the code is run against its own asserts (Python and node) and only passing code is kept."""
    from huggingface_hub import HfApi
    from tools.expand import bcode, runner

    key = os.environ["DEEPSEEK_API_KEY"]
    models = runner.list_models(key)
    model = next((m for m in models if "pro" in m), models[0])
    miss, hit, out = RATES["flash" if "flash" in model else "pro"]
    batches = bcode.plan_batches(bcode.make_rows(n, start, seed), per_batch)
    base = (Path(PROMPTS) / "deepseek_B_code.md").read_text(encoding="utf-8")
    wd = Path(DATA) / ("expand_" + tag)
    wd.mkdir(parents=True, exist_ok=True)
    budget = runner.Budget(cap, miss, hit, out)
    print(f"model {model}, {n} code tasks in {len(batches)} batches, cap ${cap}", flush=True)

    def push() -> None:
        try:
            volume.commit()
            HfApi().upload_folder(folder_path=str(wd), path_in_repo="sft_" + tag, repo_id=HF_REPO, repo_type="model",
                                  allow_patterns=["accepted_*.txt", "built_*.jsonl", "stats.json", "rejects.jsonl", "raw/*.txt"],
                                  commit_message="sft_%s: progress" % tag)
        except Exception as err:
            print("[hf] push warning:", err, flush=True)

    try:
        stats = bcode.run(batches, lambda p: runner.call_deepseek(p, key, model), wd, budget, workers=workers, base_text=base)
    finally:
        push()
    print(json.dumps(stats), flush=True)
    return stats


@app.function(image=image, cpu=1, memory=1024, timeout=12 * 3600 + 900, secrets=[ds_secret, hf_secret])
def code_pipeline(n: int, cap: float, workers: int, tag: str, start: int, seed: int) -> dict:
    code_, tail = selftest.remote()
    print(tail, flush=True)
    if code_ != 0:
        raise RuntimeError("selftest failed; nothing was spent. See the log above.")
    return code.remote(n, cap, workers, tag, start, 10, seed)


@app.local_entrypoint()
def start_code_pilot(n: int = 40, cap: float = 2.0, workers: int = 4) -> None:
    """modal run --detach deploy/modal_expand.py::start_code_pilot   (40 code tasks on Pro; folder sft_code_pilot/)"""
    call = code_pipeline.spawn(n, cap, workers, "code_pilot", 1, 7)
    print("spawned pipeline:", call.object_id)


@app.local_entrypoint()
def start_code_full(n: int = 2400, cap: float = 8.0, workers: int = 16) -> None:
    """modal run --detach deploy/modal_expand.py::start_code_full   (2,400 code tasks on Pro, from id 101; folder sft_code_full/)"""
    call = code_pipeline.spawn(n, cap, workers, "code_full", 101, 21)
    print("spawned pipeline:", call.object_id)


@app.function(image=image, cpu=2, memory=4096, timeout=3600, volumes={DATA: volume}, secrets=[hf_secret])
def recheck_b_d() -> dict:
    """Free: re-read the saved raw replies of the code full run and the thinking pilot with the fixed reader (examples in
    separate text blocks, kinder arithmetic check). Writes sft_code_full_rechecked/ and sft_traces_pilot_rechecked/."""
    from huggingface_hub import HfApi
    from tools.expand import bcode, dtraces

    volume.reload()
    out = {}
    wd = Path(DATA) / "expand_code_full_v2"
    out["code"] = bcode.reprocess(Path(DATA) / "expand_code_full" / "raw", wd)
    wd2 = Path(DATA) / "expand_traces_pilot_v2"
    probs = dtraces.make_problems(20, 1)
    out["traces"] = dtraces.reprocess(dtraces.plan_batches(probs, 5), Path(DATA) / "expand_traces_pilot" / "raw", wd2)
    print(json.dumps(out), flush=True)
    volume.commit()
    api = HfApi()
    for folder, repo_dir in ((wd, "sft_code_full_rechecked"), (wd2, "sft_traces_pilot_rechecked")):
        api.upload_folder(folder_path=str(folder), path_in_repo=repo_dir, repo_id=HF_REPO, repo_type="model",
                          allow_patterns=["accepted_*.txt", "built_*.jsonl", "stats.json", "rejects.jsonl"],
                          commit_message=repo_dir)
    return out


@app.local_entrypoint()
def redo_b_d() -> None:
    """modal run deploy/modal_expand.py::redo_b_d   (free)"""
    print(recheck_b_d.remote())


@app.local_entrypoint()
def start_traces_full(n: int = 900, cap: float = 8.0, workers: int = 16) -> None:
    """modal run --detach deploy/modal_expand.py::start_traces_full   (900 thinking traces on Pro, problem ids from 21; folder sft_traces_full/)"""
    call = traces_pipeline.spawn(n, cap, workers, "traces_full", 21)
    print("spawned pipeline:", call.object_id)


@app.function(image=image, cpu=2, memory=4096, timeout=12 * 3600, volumes={DATA: volume}, secrets=[ds_secret, hf_secret])
def think(n: int, cap: float, workers: int, tag: str = "think_full", start: int = 1, per_batch: int = 8, seed: int = 31) -> dict:
    """C thinking examples on DeepSeek Pro; the python <verify> snippet must reproduce the answer."""
    from huggingface_hub import HfApi
    from tools.expand import cthink, runner

    key = os.environ["DEEPSEEK_API_KEY"]
    models = runner.list_models(key)
    model = next((m for m in models if "pro" in m), models[0])
    miss, hit, out = RATES["flash" if "flash" in model else "pro"]
    batches = cthink.plan_batches(cthink.make_rows(n, start, seed), per_batch)
    base = (Path(PROMPTS) / "deepseek_C_thinking.md").read_text(encoding="utf-8")
    wd = Path(DATA) / ("expand_" + tag)
    wd.mkdir(parents=True, exist_ok=True)
    budget = runner.Budget(cap, miss, hit, out)
    print(f"model {model}, {n} thinking tasks in {len(batches)} batches, cap ${cap}", flush=True)
    try:
        stats = cthink.run(batches, lambda p: runner.call_deepseek(p, key, model), wd, budget, workers=workers, base_text=base)
    finally:
        try:
            volume.commit()
            HfApi().upload_folder(folder_path=str(wd), path_in_repo="sft_" + tag, repo_id=HF_REPO, repo_type="model",
                                  allow_patterns=["accepted_*.txt", "built_*.jsonl", "stats.json", "rejects.jsonl", "raw/*.txt"],
                                  commit_message="sft_%s: progress" % tag)
        except Exception as err:
            print("[hf] push warning:", err, flush=True)
    print(json.dumps(stats), flush=True)
    return stats


@app.function(image=image, cpu=1, memory=1024, timeout=12 * 3600 + 900, secrets=[ds_secret, hf_secret])
def think_pipeline(n: int, cap: float, workers: int, tag: str, start: int) -> dict:
    code_, tail = selftest.remote()
    print(tail, flush=True)
    if code_ != 0:
        raise RuntimeError("selftest failed; nothing was spent. See the log above.")
    return think.remote(n, cap, workers, tag, start)


@app.local_entrypoint()
def start_think_full(n: int = 1800, cap: float = 8.0, workers: int = 16) -> None:
    """modal run --detach deploy/modal_expand.py::start_think_full   (1,800 thinking examples on Pro; folder sft_think_full/)"""
    call = think_pipeline.spawn(n, cap, workers, "think_full", 1)
    print("spawned pipeline:", call.object_id)




@app.function(image=proj_image, cpu=4, memory=8192, timeout=12 * 3600, volumes={DATA: volume}, secrets=[ds_secret, hf_secret])
def projects(n: int, cap: float, workers: int, tag: str = "proj_full", start: int = 2101, per_batch: int = 6, seed: int = 53) -> dict:
    """P projects on DeepSeek Pro: every skeleton is run for real (compile, run, fix) and only passing ones are kept."""
    from huggingface_hub import HfApi
    from tools.expand import runner, skel

    key = os.environ["DEEPSEEK_API_KEY"]
    models = runner.list_models(key)
    model = next((m for m in models if "pro" in m), models[0])
    miss, hit, out = RATES["flash" if "flash" in model else "pro"]
    batches = skel.plan_batches(skel.make_rows(n, start, seed), per_batch)
    base = (Path(PROMPTS) / "deepseek_P_projects_2001-2024.md").read_text(encoding="utf-8").split("## 4. Work plan")[0]
    wd = Path(DATA) / ("expand_" + tag)
    wd.mkdir(parents=True, exist_ok=True)
    budget = runner.Budget(cap, miss, hit, out)
    print(f"model {model}, {n} projects in {len(batches)} batches, cap ${cap}", flush=True)
    try:
        stats = skel.run(batches, lambda p: runner.call_deepseek(p, key, model), wd, budget, workers=workers, base_text=base)
    finally:
        try:
            volume.commit()
            HfApi().upload_folder(folder_path=str(wd), path_in_repo="sft_" + tag, repo_id=HF_REPO, repo_type="model",
                                  allow_patterns=["accepted_*.txt", "built_*.jsonl", "stats.json", "rejects.jsonl", "raw/*.txt"],
                                  commit_message="sft_%s: progress" % tag)
        except Exception as err:
            print("[hf] push warning:", err, flush=True)
    print(json.dumps(stats), flush=True)
    return stats


@app.function(image=image, cpu=1, memory=1024, timeout=12 * 3600 + 900, secrets=[ds_secret, hf_secret])
def projects_pipeline(n: int, cap: float, workers: int, tag: str, start: int) -> dict:
    code_, tail = selftest.remote()
    print(tail, flush=True)
    if code_ != 0:
        raise RuntimeError("selftest failed; nothing was spent. See the log above.")
    return projects.remote(n, cap, workers, tag, start)


@app.local_entrypoint()
def start_projects_full(n: int = 800, cap: float = 8.0, workers: int = 12) -> None:
    """modal run --detach deploy/modal_expand.py::start_projects_full   (800 real-sandbox projects on Pro; folder sft_proj_full/)"""
    call = projects_pipeline.spawn(n, cap, workers, "proj_full", 2101)
    print("spawned pipeline:", call.object_id)
