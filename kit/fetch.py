"""Download the files in /mnt/.plan.json from the Hugging Face repo into /mnt.

  python /opt/kit/fetch.py [--plan /mnt/.plan.json] [--workers 3] [--verify]
  python /opt/kit/fetch.py --source /some/local/tree     # copy from a local mirror instead (testing)

Files already present at the right size are skipped, so a re-run resumes.
--verify also checks each file's SHA-256 against the manifest (slow: it
reads everything once more). HF_TOKEN is used if the repo is gated.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

KIT = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="/mnt/.plan.json")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--source", default="", help="copy from this local tree (laid out like /mnt) instead of downloading")
    a = ap.parse_args()
    cfg = json.load(open(f"{KIT}/config.json"))
    files = json.load(open(a.plan))["files"]
    todo = [f for f in files if not (os.path.exists(f"/mnt/{f['path']}") and os.path.getsize(f"/mnt/{f['path']}") == f["bytes"])]
    print(f"{len(files) - len(todo)} of {len(files)} files already here; fetching {len(todo)} "
          f"({sum(f['bytes'] for f in todo) / 1e9:.1f} GB) from {a.source or 'huggingface.co/' + cfg['hf_repo']}", flush=True)

    def one(f):
        dest = f"/mnt/{f['path']}"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if a.source:
            shutil.copyfile(os.path.join(a.source, f["path"]), dest + ".part")
            os.replace(dest + ".part", dest)
        else:
            from huggingface_hub import hf_hub_download
            hf_hub_download(cfg["hf_repo"], filename=f["path"], local_dir="/mnt", token=os.environ.get("HF_TOKEN") or None)
        if os.path.getsize(dest) != f["bytes"]:
            raise RuntimeError(f"{f['path']}: got {os.path.getsize(dest)} bytes, expected {f['bytes']}")
        print(f"  done  {f['bytes'] / 1e9:6.2f} GB  {f['path']}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    if a.verify:
        bad = [f["path"] for f in files if sha256(f"/mnt/{f['path']}") != f["sha256"]]
        if bad:
            sys.exit("Checksum mismatch (delete these and run again): " + ", ".join(bad))
        print("All checksums match.")


if __name__ == "__main__":
    main()
