"""Zip one card's results for sending back.

  python /opt/kit/package.py --gpu rtx4090-3fa1 [--out /results]

Takes the result tree fit.py wrote (/mnt/flash/downloads/_quant_bench/fit/<id>):
card.json, every result.json, and each render as a 512 px JPEG instead of the
full PNG, so the zip stays a few MB. Nothing else from the machine goes in;
card.json holds what the bench recorded (GPU, driver, RAM, PCIe link, power
limit) plus the contributor block run.sh wrote (credit name, OS, CPU).
"""

import argparse
import glob
import io
import json
import os
import sys
import zipfile

from PIL import Image

sys.path.insert(0, "/opt/bench")
import lineup as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--out", default="/results")
    a = ap.parse_args()
    root = f"{L.ROOT}/fit/{a.gpu}"
    results = sorted(glob.glob(f"{root}/*/*/result.json"))
    if not results:
        sys.exit(f"No results under {root} yet.")
    done = [r for r in results if not json.load(open(r)).get("error")]
    os.makedirs(a.out, exist_ok=True)
    out = f"{a.out}/qbench-{a.gpu}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [f"{root}/card.json", *results, *sorted(glob.glob(f"{root}/_te/*.json"))]:
            if os.path.exists(p):
                z.write(p, f"fit/{a.gpu}/{os.path.relpath(p, root)}")
        for png in sorted(glob.glob(f"{root}/*/*/p0.png")):
            im = Image.open(png).convert("RGB")
            im.thumbnail((512, 512))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
            z.writestr(f"fit/{a.gpu}/{os.path.relpath(png, root)[:-4]}.jpg", buf.getvalue())
    print(f"{len(done)} of {len(results)} runs finished. Results: results/{os.path.basename(out)} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
