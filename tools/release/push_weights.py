"""Copy a trained checkpoint inside the Hugging Face model repo, server-side (no download, no re-upload).

Never reads a token: huggingface_hub uses the login the owner made himself.
"""
import sys

REPO = "bekhruzsuleyman/alethic-300m"
SRC = "runs/sft3c/best"
DST = "sft/sft3c"


class PushError(Exception):
    pass


def push(api, repo, src, dst):
    if not dst or dst.strip("/") == "":
        raise PushError("destination must be a subfolder, never the repo root")
    src = src.rstrip("/") + "/"
    dst = dst.rstrip("/") + "/"
    names = sorted(f for f in api.list_repo_files(repo, repo_type="model") if f.startswith(src))
    if not names:
        raise PushError("nothing found under %s in %s" % (src, repo))
    from huggingface_hub import CommitOperationCopy
    ops = [CommitOperationCopy(src_path_in_repo=n, path_in_repo=dst + n[len(src):]) for n in names]
    api.create_commit(repo_id=repo, operations=ops, repo_type="model",
                      commit_message="Add SFT-3c weights in %s (base files untouched)" % dst)
    return [o.path_in_repo for o in ops]


def main():
    from huggingface_hub import HfApi
    done = push(HfApi(), REPO, SRC, DST)
    print("Copied %d files to https://huggingface.co/%s/tree/main/%s" % (len(done), REPO, DST.strip("/")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
