"""Run one shell command in a working directory with a fixed locale, a timeout and an output cap."""
import os
import re
import shutil
import subprocess

MARK = "\n[truncated]"
TIMING = re.compile(r"\bin \d+(?:\.\d+)?s\b")  # "Ran 11 tests in 0.002s": varies per run, so fixed
NET_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "NO_PROXY", "no_proxy",
            "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "PIP_CERT")


TOOLCHAINS = ("rustc", "cargo", "go", "gcc", "node", "tsc", "npm")


def _path():
    """Fixed base PATH plus the folders of the language toolchains found on the builder's own PATH."""
    dirs = ["/usr/local/bin", "/usr/bin", "/bin"]
    for tool in TOOLCHAINS:
        found = shutil.which(tool)
        if found and os.path.dirname(found) not in dirs:
            dirs.append(os.path.dirname(found))
    return ":".join(dirs)


def _toolchain_env():
    home = os.path.expanduser("~")
    return {"RUSTUP_HOME": os.environ.get("RUSTUP_HOME", os.path.join(home, ".rustup")),
            "CARGO_HOME": os.environ.get("CARGO_HOME", os.path.join(home, ".cargo")),
            "GOCACHE": "/tmp/sftbuild_gocache", "GOPATH": "/tmp/sftbuild_gopath", "GOTOOLCHAIN": "local",
            "GOFLAGS": "-mod=mod",
            "XDG_CONFIG_HOME": "/tmp/sftbuild_xdg_config", "XDG_CACHE_HOME": "/tmp/sftbuild_xdg_cache"}  # caches live outside the working directory, so `ls` stays clean


def run(cmd: str, cwd: str, timeout: int = 10, cap: int = 100_000, network: bool = False):
    env = {"PATH": _path(), "LC_ALL": "C", "HOME": cwd,
           "PYTHONDONTWRITEBYTECODE": "1", **_toolchain_env()}  # no __pycache__: stale bytecode after same-size edits, and noise in ls
    if network:  # proxy settings only for steps that need the allowlisted network
        env.update({k: os.environ[k] for k in NET_VARS if k in os.environ})
    try:
        r = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return 124, ""
    out = TIMING.sub("in 0.001s", (r.stdout + r.stderr).strip())
    if len(out) > cap:
        out = out[:cap] + MARK
    return r.returncode, out


def fresh_dir(path: str) -> None:
    """Empty and recreate a working directory, but only one this builder made earlier
    (a sibling marker file proves it). Raises ValueError for any other non-empty directory."""
    marker = path.rstrip("/") + ".sftbuild_marker"
    if os.path.isdir(path) and os.listdir(path) and not os.path.exists(marker):
        raise ValueError("workdir %s is not empty and was not created by sftbuild; refusing to delete it" % path)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    with open(marker, "w") as f:
        f.write("created by sftbuild\n")
