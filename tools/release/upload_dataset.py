"""Upload the alethic-newborn shards and the dataset card to Hugging Face, then turn the access gate on.

Never reads or stores a token: huggingface_hub uses the login the owner made himself.
"""
import glob
import os
import sys

REPO = "bekhruzsuleyman/alethic-newborn"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARD_PATH = os.path.join(ROOT, "docs", "release", "alethic-newborn-README.md")


class UploadError(Exception):
    pass


def files_to_upload(dataset_dir, card_path=CARD_PATH):
    shards = sorted(glob.glob(os.path.join(dataset_dir, "alethic-newborn0*.jsonl"))
                    + glob.glob(os.path.join(dataset_dir, "alethic-newborn0*.parquet")))
    manifest = os.path.join(dataset_dir, "alethic-newborn.manifest.json")
    if not shards or not os.path.exists(manifest):
        raise UploadError("no shards or manifest in %s: run 32_make_public_copy.bat first" % dataset_dir)
    if not os.path.exists(card_path):
        raise UploadError("dataset card not found: %s" % card_path)
    out = [(card_path, "README.md"), (manifest, os.path.basename(manifest))]
    out += [(p, os.path.basename(p)) for p in shards]
    return out


def upload(api, repo, dataset_dir, card_path=CARD_PATH):
    files = files_to_upload(dataset_dir, card_path)
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    for path, name in files:
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo, repo_type="dataset")
    api.update_repo_settings(repo_id=repo, repo_type="dataset", gated="auto")
    return [n for _, n in files]


def main(argv):
    from huggingface_hub import HfApi
    names = upload(HfApi(), REPO, argv[1] if len(argv) > 1 else "dataset")
    print("Uploaded %d files to https://huggingface.co/datasets/%s" % (len(names), REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
