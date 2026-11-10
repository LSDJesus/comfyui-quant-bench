"""List the files fit.py needs on a card, for copying them to that machine.

  python3 stage.py --vram 16 --blackwell > files.txt     # paths relative to /
  rsync -a --files-from=files.txt / host:/stage/root/

Run on the main server (plain python3, no torch). The list covers the model
files fit.py will load for a card with that much VRAM (lineup.fit_variants),
the conditioning and p0 references, each family's text encoder and VAE, and
the SDXL checkpoints (their CLIP) with the extracted VAEs. Code (the bench
directory, the ComfyUI checkout and its venv) is copied separately; see the
README. Mount the staged tree at / in the container (/mnt/flash, /mnt/cold,
/mnt/working), so every path in lineup.py resolves unchanged.

Models on a local SSD also matter for what is measured: ComfyUI picks how it
offloads from where the file lives (comfy.storage.fast_storage), streaming
from a fast local disk and holding the model in pinned RAM otherwise. Most
users keep models on a local SSD, so the fit pass stages there.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lineup as L  # noqa: E402


def files(vram_gb, blackwell, families=None, fp8=False):
    out = []
    seen = set()

    def add(p):
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    for mid in L.FIT_MODELS:
        m = L.MODELS[mid]
        if families and m["family"] not in families:
            continue
        for v in L.fit_variants(mid, vram_gb * 1e9, blackwell, fp8=fp8):
            add(L.file_for(mid, v))
        fam = L.FAMILIES[m["family"]]
        add(L.cond_path(mid))
        add(L.ref_path(mid, 0))
        add(fam["te"][0] if fam["te"] else None)
        add(fam["vae"])
        if "ckpt" in m:
            add(m["ckpt"])
            add(f"{L.SRC}/{os.path.basename(m['ckpt']).rsplit('.', 1)[0]}_vae.safetensors")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vram", type=float, required=True, help="card VRAM in GB")
    ap.add_argument("--blackwell", action="store_true")
    ap.add_argument("--fp8", action="store_true", help="RTX 40 (Ada): add fp8 scaled")
    ap.add_argument("--families", default="")
    ap.add_argument("--sizes", action="store_true", help="print sizes and a total instead")
    args = ap.parse_args()
    fs = files(args.vram, args.blackwell, [f for f in args.families.split(",") if f], fp8=args.fp8)
    missing = [f for f in fs if not os.path.exists(f)]
    if missing:
        sys.exit("missing: " + ", ".join(missing))
    if args.sizes:
        tot = 0
        for f in fs:
            s = os.path.getsize(f)
            tot += s
            print(f"{s / 1e9:7.2f} GB  {f}")
        print(f"{tot / 1e9:7.1f} GB total, {len(fs)} files")
        return
    for f in fs:
        print(f.lstrip("/"))


if __name__ == "__main__":
    main()
