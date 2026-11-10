"""Benchmark prep (PRO 6000, once): SDXL UNet extraction, DMD2 merges,
conditioning.

  python prep.py [unets|merges|cond|all]

* unets  — each SDXL checkpoint's UNet as a bare-key diffusion-model file
           (the bakers and UNETLoader both read that), plus its VAE.
* merges — base + DMD2 LoRA, merged at full precision, saved as a new source
           (the "turbo" row for MOSS and SDXL).
* cond   — every prompt + negative, encoded ONCE with the family's text
           encoder (SDXL: each checkpoint's own CLIP) and saved, so every
           variant on both GPUs samples from identical conditioning.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import lineup as L  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

UNET_PREFIX = "model.diffusion_model."
VAE_PREFIX = "first_stage_model."


def vae_path_for(ckpt):
    return f"{L.SRC}/{os.path.basename(ckpt).rsplit('.', 1)[0]}_vae.safetensors"


def extract(ckpt, unet_dst):
    if os.path.exists(unet_dst) and os.path.exists(vae_path_for(ckpt)):
        C.log("have", unet_dst)
        return
    os.makedirs(L.SRC, exist_ok=True)
    unet, vae = {}, {}
    with safe_open(ckpt, framework="pt") as f:
        for k in f.keys():
            if k.startswith(UNET_PREFIX):
                unet[k[len(UNET_PREFIX):]] = f.get_tensor(k)
            elif k.startswith(VAE_PREFIX):
                vae[k[len(VAE_PREFIX):]] = f.get_tensor(k)
    dt = {str(v.dtype) for v in unet.values()}
    C.log(f"{os.path.basename(ckpt)}: UNet {len(unet)} tensors {dt}, VAE {len(vae)}")
    save_file(unet, unet_dst, metadata={"luna_bench_source": os.path.basename(ckpt)})
    save_file(vae, vae_path_for(ckpt))


def do_unets():
    seen = set()
    for mid, m in L.MODELS.items():
        if m["family"] == "sdxl" and "merge" not in m and m.get("kind") != "turbo_lora":
            if m["src"] not in seen:
                extract(m["ckpt"], m["src"])
                seen.add(m["src"])


def do_merges():
    for mid, m in L.MODELS.items():
        if "merge" not in m:
            continue
        if os.path.exists(m["src"]):
            C.log("have", m["src"])
            continue
        base_id, lora, strength = m["merge"]
        base = L.MODELS[base_id]["src"]
        C.log(f"merge {mid}: {os.path.basename(base)} + {os.path.basename(lora)} @ {strength}")
        with safe_open(base, framework="pt") as f:
            src_dtype = {f.get_slice(k).get_dtype() for k in list(f.keys())[:50]}
        model = C.load_diffusion(base)
        model = C.apply_lora(model, lora, strength)
        model.patch_model(device_to=torch.device("cuda"))
        sd = model.model.diffusion_model.state_dict()
        n_lora = len(model.patches)
        out = {}
        for k, v in sd.items():
            v = v.detach()
            if v.is_floating_point() and v.dtype != torch.float32:
                v = v.to(torch.bfloat16 if "BF16" in src_dtype else torch.float16)
            out[k] = v.cpu().contiguous()
        model.unpatch_model()
        # Keys the module does not save (MOSS's `rope.bands` buffer) are
        # copied from the source, so the merge is the source's key set.
        with safe_open(base, framework="pt") as f:
            for k in f.keys():
                if k not in out:
                    out[k] = f.get_tensor(k)
                    C.log(f"  kept from source: {k}")
        C.log(f"  {n_lora} patched weights, saving {len(out)} tensors -> {m['src']}")
        save_file(out, m["src"], metadata={"luna_bench_merge": f"{os.path.basename(base)}+{os.path.basename(lora)}@{strength}"})
        del model, sd, out
        C.free_all()


def do_cond():
    os.makedirs(f"{L.ROOT}/cond", exist_ok=True)
    done = set()
    for mid, m in L.MODELS.items():
        path = L.cond_path(mid)
        if path in done or os.path.exists(path):
            done.add(path)
            continue
        fam = L.FAMILIES[m["family"]]
        if m["family"] == "sdxl":
            _, clip, _, _ = comfy_ckpt_clip(m["ckpt"])
        else:
            clip = C.load_te(*fam["te"])
        pos = [C.encode(clip, p) for p in fam["prompts"]]
        neg = C.encode(clip, L.NEG[m["family"]])
        torch.save({"pos": pos, "neg": neg, "prompts": fam["prompts"], "negative": L.NEG[m["family"]],
                    "te": fam["te"][0] if fam["te"] else m["ckpt"]}, path)
        C.log(f"cond {path}: {[tuple(c[0][0].shape) for c in pos]}")
        done.add(path)
        del clip
        C.free_all()


def comfy_ckpt_clip(ckpt):
    import comfy.sd
    out = comfy.sd.load_checkpoint_guess_config(ckpt, output_vae=False, output_clip=True, output_model=False)
    return out[0], out[1], out[2], None


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("unets", "all"):
        do_unets()
    if what in ("merges", "all"):
        do_merges()
    if what in ("cond", "all"):
        do_cond()
    C.log("prep done:", what)
