"""Work out what this card will run and which files that needs.

  python /opt/kit/plan.py [--families zimage,sdxl] [--out /mnt/.plan.json]

Runs inside the container, on the GPU run.sh picked. The formats per model
come from the bench's own rule (lineup.fit_variants): int8 and w4a8 on every
card, fp8 scaled on RTX 40 (fp8 hardware, no fp4), nvfp4 on RTX 50, and the
bf16 original when the file is under 16 GB or fits the card. Sizes come from
the manifest, since nothing is downloaded yet. Text encoders and SDXL
checkpoints are left out: the kit runs fit.py with --no-te (the prompts are
pre-encoded in the conditioning files).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "/opt/bench")
import lineup as L  # noqa: E402

KIT = os.path.dirname(os.path.abspath(__file__))


def card():
    import torch
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible inside the container. Is the NVIDIA Container Toolkit installed?")
    p = torch.cuda.get_device_properties(0)
    cap = tuple(torch.cuda.get_device_capability(0))
    return {"name": p.name, "vram": p.total_memory, "capability": cap,
            "blackwell": cap[0] >= 10, "fp8": cap == (8, 9)}


def plan(c, families):
    man = json.load(open(f"{KIT}/manifest.json"))
    by_path = {f"/mnt/{f['path']}": f for f in man["files"]}
    size = lambda p: by_path[p]["bytes"] if p in by_path else None  # noqa: E731
    want, runs = {}, []
    for mid in L.FIT_MODELS:
        m = L.MODELS[mid]
        if families and m["family"] not in families:
            continue
        vs = L.fit_variants(mid, c["vram"], c["blackwell"], fp8=c["fp8"], bf16_bytes=size(L.file_for(mid, "bf16")))
        for v in vs:
            want[L.file_for(mid, v)] = 1
            runs.append([mid, v])
        fam = L.FAMILIES[m["family"]]
        for p in (L.cond_path(mid), L.ref_path(mid, 0), fam["vae"]):
            if p:
                want[p] = 1
        if "ckpt" in m:   # SDXL decodes with the VAE extracted from its checkpoint
            want[f"{L.SRC}/{os.path.basename(m['ckpt']).rsplit('.', 1)[0]}_vae.safetensors"] = 1
    missing = [p for p in want if p not in by_path]
    if missing:
        sys.exit("The manifest lacks: " + ", ".join(missing))
    files = [by_path[p] for p in want]
    have = [f for f in files if os.path.exists(f"/mnt/{f['path']}") and os.path.getsize(f"/mnt/{f['path']}") == f["bytes"]]
    return {"card": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()}, "families": families,
            "runs": runs, "files": files, "bytes": sum(f["bytes"] for f in files),
            "bytes_present": sum(f["bytes"] for f in have)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="")
    ap.add_argument("--out", default="/mnt/.plan.json")
    a = ap.parse_args()
    fams = [f for f in a.families.split(",") if f]
    bad = [f for f in fams if f not in L.FAMILIES]
    if bad:
        sys.exit(f"Unknown family {bad}; pick from {', '.join(L.FAMILIES)}")
    c = card()
    p = plan(c, fams)
    with open(a.out, "w") as f:
        json.dump(p, f, indent=1)
    gen = "Blackwell" if c["blackwell"] else "Ada (RTX 40)" if c["fp8"] else f"compute {c['capability'][0]}.{c['capability'][1]}"
    print(f"Card: {c['name']}, {c['vram'] / 2**30:.0f} GB, {gen}")
    by_model = {}
    for mid, v in p["runs"]:
        by_model.setdefault(mid, []).append(v)
    for mid, vs in by_model.items():
        print(f"  {L.MODELS[mid]['name']:<44} {', '.join(vs)}")
    need = p["bytes"] - p["bytes_present"]
    with open(os.path.join(os.path.dirname(a.out), ".plan_need"), "w") as f:   # for run.sh, which may lack python
        f.write(str(need))
    print(f"{len(p['runs'])} runs. Files: {p['bytes'] / 1e9:.0f} GB, {p['bytes_present'] / 1e9:.0f} GB already here, "
          f"{need / 1e9:.0f} GB to download.")


if __name__ == "__main__":
    main()
