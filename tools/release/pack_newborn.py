"""Pack the built SFT data into numbered shards named alethic-newborn0001, 0002, ...

    python -m tools.release.pack_newborn BUILT_DIR OUT_DIR [--per-shard N] [--seed S] [--no-parquet]

Every *.jsonl in BUILT_DIR is read, all examples are shuffled together with a
fixed seed (so every shard mixes every kind), cut into shards of N examples and
written as alethic-newborn0001.jsonl (+ .parquet when pyarrow is installed).
Ids are kept as they are, so the trainer's split by id still works.
A manifest lists every shard, its example count and the kinds inside.
"""
import json
import os
import random
import re
import sys

NAME = "alethic-newborn"


NAME_FIXES = [(re.compile(r" ?\(Bekhruz Suleyman\)"), ""), (re.compile(r"Bekhruz Suleyman"), "Bekhruz Suleyman"),
              (re.compile(r"Bexruzbek"), "Bekhruz"), (re.compile(r"Suleymanov"), "Suleyman")]


def scrub(text):
    """The legal name on his ID is not released: the model's maker is just 'Bekhruz Suleyman'."""
    for pat, new in NAME_FIXES:
        text = pat.sub(new, text)
    return text


def _read(built_dir):
    rows = []
    for fn in sorted(os.listdir(built_dir)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(built_dir, fn), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(scrub(line)))
    return rows


def _write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        "id": [r["id"] for r in rows],
        "kind": [r.get("kind", "") for r in rows],
        "lang": [r.get("lang", "") for r in rows],
        "segments": [json.dumps(r["segments"], ensure_ascii=False) for r in rows],
    })
    pq.write_table(table, path)


def pack(built_dir, out_dir, per_shard=2500, seed=1, parquet=True):
    rows = _read(built_dir)
    random.Random(seed).shuffle(rows)
    os.makedirs(out_dir, exist_ok=True)
    have_parquet = False
    if parquet:
        try:
            import pyarrow  # noqa: F401
            have_parquet = True
        except ImportError:
            have_parquet = False
    files = []
    kinds = {}
    for i in range(0, len(rows), per_shard):
        chunk = rows[i:i + per_shard]
        stem = "%s%04d" % (NAME, i // per_shard + 1)
        with open(os.path.join(out_dir, stem + ".jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            for r in chunk:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        if have_parquet:
            _write_parquet(os.path.join(out_dir, stem + ".parquet"), chunk)
        k = {}
        for r in chunk:
            kk = r.get("kind", "")
            k[kk] = k.get(kk, 0) + 1
            kinds[kk] = kinds.get(kk, 0) + 1
        files.append({"file": stem + ".jsonl", "examples": len(chunk), "kinds": k})
    manifest = {"name": NAME, "examples": len(rows), "shards": len(files),
                "seed": seed, "parquet": have_parquet, "kinds": kinds, "files": files}
    with open(os.path.join(out_dir, NAME + ".manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    return manifest


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    opts = argv[1:]
    per, seed = 2500, 1
    if "--per-shard" in opts:
        per = int(opts[opts.index("--per-shard") + 1])
        args.remove(opts[opts.index("--per-shard") + 1])
    if "--seed" in opts:
        seed = int(opts[opts.index("--seed") + 1])
        args.remove(opts[opts.index("--seed") + 1])
    if len(args) != 2:
        print(__doc__)
        return 2
    m = pack(args[0], args[1], per_shard=per, seed=seed, parquet="--no-parquet" not in opts)
    print("packed %d examples into %d shards (%s0001 ... %s%04d), parquet=%s" %
          (m["examples"], m["shards"], NAME, NAME, m["shards"], m["parquet"]))
    for k, v in sorted(m["kinds"].items(), key=lambda kv: -kv[1]):
        print("  %-12s %d" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
