"""Start the agent with no arguments: weights, tokenizer and config are found next to the program (used by the .exe)."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

TOKENIZER = os.path.join("data", "tokenizer", "alethic-tokenizer-v1", "alethic-tokenizer-v1.model")
REPO, WEIGHTS_FILE = "bekhruzsuleyman/alethic-300m", "runs/sft3b/best/model.pt"
WEIGHTS_NAME = "model-sft3b.pt"  # named after the model, so an older model.pt next to the program is never loaded by mistake


class WeightsError(Exception):
    """The model weights are missing and could not be fetched; the message says what to do."""


def _can_write(folder: str) -> bool:
    if not os.path.isdir(folder):
        return True
    try:
        with tempfile.TemporaryFile(dir=folder):
            return True
    except OSError:
        return False


def user_folder() -> str:
    return os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "alethic-agent")


def app_dirs(frozen: bool, exe: str, meipass: str | None, source: str, writable=None, fallback: str | None = None):
    """(bundle folder with the tokenizer and config, data folder for model.pt and the work folder).
    Data sits next to the exe, or in the user's own folder when the exe lives somewhere read-only (Program Files)."""
    if frozen:
        here = os.path.dirname(exe)
        ok = (writable or _can_write)(here)
        return meipass or here, here if ok else (fallback or user_folder())
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(source))))
    return root, root


def build_argv(user_args: list, bundle: str, data: str) -> list:
    """Default arguments first; anything the user typed comes later, and argparse keeps the last value."""
    return ["--weights", os.path.join(data, WEIGHTS_NAME),
            "--tokenizer", os.path.join(bundle, TOKENIZER),
            "--model", os.path.join(bundle, "configs", "model_gqa.yaml"),
            "--suggest", os.path.join(data, "work")] + list(user_args)


def ensure_weights(path: str, download, say=print) -> None:
    """Fetch the weights once (download() returns a cached file path) and copy them to `path`."""
    if os.path.exists(path):
        return
    say("Downloading the SFT-3b weights from Hugging Face, 1.2 GB, once…")
    folder = os.path.dirname(path) or "."
    part = path + ".part"
    try:
        os.makedirs(folder, exist_ok=True)
        shutil.copyfile(download(), part)
        os.replace(part, path)          # a half-copied file never takes the name model.pt
    except Exception as exc:
        if os.path.exists(part):
            os.remove(part)
        raise WeightsError("Could not get the model weights (%s).\nCheck your internet connection and start the program again,\n"
                           "or copy the file %s into this folder:\n  %s" % (exc, os.path.basename(path), folder)) from exc


def _download():
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, WEIGHTS_FILE)


SELFTEST_MODULES = ["torch", "sentencepiece", "numpy", "huggingface_hub", "alethic.model.transformer", "alethic.agent.cli"]


def check_imports(names, importer=None) -> dict:
    """Import each module; return {name: error text} for those that fail (the .exe build runs this to catch a missing DLL)."""
    import importlib
    importer = importer or importlib.import_module
    failed = {}
    for name in names:
        try:
            importer(name)
        except BaseException as exc:
            failed[name] = "%s: %s" % (type(exc).__name__, exc)
    return failed


def _selftest() -> int:
    failed = check_imports(SELFTEST_MODULES)
    for name, err in failed.items():
        print("FAILED  %s  %s" % (name, err))
    print("selftest: %s" % ("all %d modules load" % len(SELFTEST_MODULES) if not failed else "%d failed" % len(failed)))
    return 1 if failed else 0


def _set_title(title: str) -> None:
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW(title)
        except Exception:
            pass


def _wait(prompt: str) -> None:
    try:
        input(prompt)
    except EOFError:
        pass


def run() -> None:
    if sys.argv[1:] == ["--selftest"]:
        raise SystemExit(_selftest())
    frozen = bool(getattr(sys, "frozen", False))
    bundle, data = app_dirs(frozen, sys.executable, getattr(sys, "_MEIPASS", None), __file__)
    os.environ.setdefault("PYTHONUTF8", "1")
    _set_title("Alethic agent")
    try:
        from alethic.agent.cli import main
        ensure_weights(os.path.join(data, WEIGHTS_NAME), _download)
        main(build_argv(sys.argv[1:], bundle, data))
    except SystemExit:
        raise
    except WeightsError as exc:
        print("\n" + str(exc))
        _wait("\nPress Enter to close.")
        return
    except BaseException:
        if not frozen:
            raise
        import traceback
        traceback.print_exc()
        _wait("\nSomething went wrong. Press Enter to close.")
        return
    if frozen:
        _wait("\nPress Enter to close.")
