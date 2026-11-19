"""Replace the body of the Hugging Face model card, keeping its existing front matter (license, tags).

The old card is saved next to the new one first. Never reads a token: huggingface_hub uses the owner's login.
"""
import os
import sys

REPO = "bekhruzsuleyman/alethic-300m"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BODY_PATH = os.path.join(ROOT, "docs", "release", "alethic-300m-card-body.md")
BACKUP_PATH = os.path.join(ROOT, "docs", "release", "alethic-300m-card-before.md")


class CardError(Exception):
    pass


def _front(text):
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[:end + 5]
    return ""


def merge_card(old, body):
    if _front(body):
        raise CardError("the new body must not contain its own front matter")
    front = _front(old)
    return (front + "\n" + body) if front else body


def main():
    from huggingface_hub import HfApi, hf_hub_download
    old = open(hf_hub_download(REPO, "README.md"), encoding="utf-8").read()
    with open(BACKUP_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(old)
    new = merge_card(old, open(BODY_PATH, encoding="utf-8").read())
    HfApi().upload_file(path_or_fileobj=new.encode("utf-8"), path_in_repo="README.md", repo_id=REPO,
                        commit_message="Update model card: SFT-3c, limits, links")
    print("Model card updated. The old card is saved in %s" % BACKUP_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
