"""Fit-and-speed pass: what runs on a given card, and how fast.

  QBENCH_DYNAMIC=1 python fit.py --gpu rtx5080 [--models id|family,...] [--variants v,...]

The consumer-card companion to run.py. Accuracy depends on the chip
generation (every Blackwell card runs the PRO 6000's kernels, every Ampere
card the 3090's), so this pass measures what changes from card to card: does
the model fit, and how long an image takes. It runs ComfyUI's default dynamic
VRAM loader (QBENCH_DYNAMIC=1, see common.py), so nothing is skipped for size:
weights that do not fit stream from system RAM, and the result records how
much of the model stayed on the card.

Scope, per card:
  models    the base and the turbo of each family (a finetune has its base's
            shape, so it fits and runs at the same speed)
  formats   int8 ConvRot and w4a8 everywhere; nvfp4 on Blackwell (sm_100+);
            bf16 as the baseline when the file is under 16 GB or fits the card
  attention PyTorch SDPA (ComfyUI's default)
  prompt    p0 only

Per (model, format):
  1. load the file (timed: disk to RAM, ComfyUI's loader)
  2. teacher-forced pass against the PRO 6000 bf16 reference: the accuracy
     check for this card, and the warm-up
  3. decode that latent (untimed), so the VAE is part of the cycle the way it
     is in a workflow
  4. timed image: sample + decode. This is the steady state a user sees on
     every image after the first: on a small card the model pages back in
     after the VAE, and that cost is in the number.

Per family, once: text encoder load + encode time (prompt and negative),
first encode (weights come from RAM) and warm.

Output: $QBENCH_ROOT/fit/<gpu>/<model>/<variant>/{result.json,p0.png},
$QBENCH_ROOT/fit/<gpu>/_te/<family>.json, $QBENCH_ROOT/fit/<gpu>/card.json.
Every TE timing and every run happens in its own process, after a CUDA
warm-up: each starts from a clean card the way a fresh ComfyUI workflow does
(an earlier model can't linger in VRAM and squeeze the next one), and a
process the kernel kills for running out of system RAM is recorded as a
crash instead of ending the pass. Finished runs are skipped; --retry-crashed
runs the crashed ones again.
"""

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import lineup as L  # noqa: E402
import prep  # noqa: E402

import torch  # noqa: E402

FIT = f"{L.ROOT}/fit"
PROMPT = 0


def run_dir(gpu, mid, v):
    return f"{FIT}/{gpu}/{mid}/{v}"


def blackwell():
    return torch.cuda.get_device_capability(0)[0] >= 10


def fp8_native():
    """Ada (RTX 40, sm_89): fp8 tensor cores, no fp4, so fp8 scaled joins its formats."""
    return tuple(torch.cuda.get_device_capability(0)) == (8, 9)


def variants_for(mid, total_vram):
    return L.fit_variants(mid, total_vram, blackwell(), fp8=fp8_native())


def meminfo():
    out = {}
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                        out[k] = round(int(v.split()[0]) * 1024 / 1e9, 2)
        else:   # Windows
            import psutil
            vm, sw = psutil.virtual_memory(), psutil.swap_memory()
            out = {"MemTotal": round(vm.total / 1e9, 2), "MemAvailable": round(vm.available / 1e9, 2),
                   "SwapTotal": round(sw.total / 1e9, 2), "SwapFree": round(sw.free / 1e9, 2)}
    except Exception as e:  # pragma: no cover
        out["error"] = str(e)
    return out


def nvml_handle():
    import pynvml
    pynvml.nvmlInit()
    idx = torch.cuda.current_device()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis and vis.split(",")[0].isdigit():
        idx = int(vis.split(",")[0])
    return pynvml, pynvml.nvmlDeviceGetHandleByIndex(idx)


def pcie_now():
    try:
        nv, h = nvml_handle()
        return {"gen": nv.nvmlDeviceGetCurrPcieLinkGeneration(h), "width": nv.nvmlDeviceGetCurrPcieLinkWidth(h)}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def gpu_state():
    """Clocks and power right now (sampled mid-sampling by run_one)."""
    try:
        nv, h = nvml_handle()
        return {"sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                "mem_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000,
                "limit_w": nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def energy_mj():
    """The card's cumulative energy counter (millijoules since driver load)."""
    try:
        nv, h = nvml_handle()
        return nv.nvmlDeviceGetTotalEnergyConsumption(h)
    except Exception:  # pragma: no cover
        return None


def card_info(gpu):
    p = f"{FIT}/{gpu}/card.json"
    info = {"gpu": gpu, "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "vram_total": torch.cuda.get_device_properties(0).total_memory,
            "host_ram": meminfo(), "torch": torch.__version__, "dynamic_vram": C.DYNAMIC,
            "platform": sys.platform, "fast_disk_forced": os.environ.get("QBENCH_FAST_DISK") == "1"}
    try:
        nv, h = nvml_handle()
        info["pcie_max"] = {"gpu_gen": nv.nvmlDeviceGetGpuMaxPcieLinkGeneration(h),
                            "link_gen": nv.nvmlDeviceGetMaxPcieLinkGeneration(h),
                            "link_width": nv.nvmlDeviceGetMaxPcieLinkWidth(h)}
        info["driver"] = nv.nvmlSystemGetDriverVersion()
        info["power_limit_w"] = nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000
        info["power_default_w"] = nv.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000
        info["display_active"] = bool(nv.nvmlDeviceGetDisplayActive(h))
    except Exception as e:  # pragma: no cover
        info["nvml_error"] = str(e)
    # The community kit describes the contributor's machine (name, OS, CPU, screens) in a JSON file.
    kp = os.environ.get("QBENCH_CONTRIBUTOR")
    if kp and os.path.exists(kp):
        with open(kp) as f:
            info["contributor"] = {**json.load(f), "display_active": info.get("display_active")}
    C.write_json(p, info)
    return info


def residency(patcher):
    try:
        return {"loaded": int(patcher.loaded_size()), "size": int(patcher.model_size())}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


# ── text encoder ────────────────────────────────────────────────────────
def te_timing(gpu, fam_id):
    p = f"{FIT}/{gpu}/_te/{fam_id}.json"
    if os.path.exists(p):
        return
    fam = L.FAMILIES[fam_id]
    C.free_all()
    out = {"family": fam_id, "gpu": gpu, "host_ram_before": meminfo(), "gpu_before": C.gpu_snapshot()}
    try:
        t0 = time.perf_counter()
        if fam_id == "sdxl":
            ckpt = L.MODELS["sdxl_base"]["ckpt"]
            clip = prep.comfy_ckpt_clip(ckpt)[1]
            out["file"], out["file_bytes"] = ckpt, None    # CLIP-L/G inside the full checkpoint
        else:
            clip = C.load_te(*fam["te"])
            out["file"], out["file_bytes"] = fam["te"][0], os.path.getsize(fam["te"][0])
        out["load_s"] = time.perf_counter() - t0
        prompt, neg = fam["prompts"][PROMPT], L.NEG[fam_id]

        def enc():
            C.encode(clip, prompt)
            C.encode(clip, neg)
        # first: one-time tokenizer/kernel setup + upload. reload: the encoder
        # evicted to RAM first, which is what a new prompt costs once the
        # diffusion model has taken the card. warm: already resident.
        _, out["encode_first_s"] = timed(enc)
        C.mm.unload_all_models()
        C.mm.soft_empty_cache(True)
        _, out["encode_reload_s"] = timed(enc)
        _, out["encode_warm_s"] = timed(enc)
        out["residency"] = residency(clip.patcher)
        out["gpu_after"] = C.gpu_snapshot()
        C.log(f"  TE {fam_id}: load {out['load_s']:.1f}s, encode first {out['encode_first_s']:.2f}s, "
              f"reload {out['encode_reload_s']:.2f}s, warm {out['encode_warm_s']:.2f}s, resident {out['residency']}")
        del clip
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        C.log(f"  TE {fam_id}: ERROR {out['error']}")
    C.free_all()
    C.write_json(p, out)


# ── one (model, format) ─────────────────────────────────────────────────
def run_one(gpu, mid, variant):
    d = run_dir(gpu, mid, variant)
    res_p = f"{d}/result.json"
    if os.path.exists(res_p):
        return
    m = L.MODELS[mid]
    fam = L.FAMILIES[m["family"]]
    path = L.file_for(mid, variant)
    base = {"model": mid, "model_name": m["name"], "family": m["family"], "kind": m["kind"],
            "variant": variant, "gpu": gpu, "gpu_name": torch.cuda.get_device_name(0),
            "file": path, "sampler": m["sampler"], "scheduler": m["scheduler"], "steps": m["steps"],
            "cfg": m["cfg"], "width": L.WIDTH, "height": L.HEIGHT, "prompt": PROMPT, "dynamic_vram": C.DYNAMIC}
    if not os.path.exists(path):
        C.log(f"SKIP {mid}/{variant}: no file {path}")
        return
    r = {**base, "file_bytes": os.path.getsize(path), "host_ram_before": meminfo(), "gpu_before": C.gpu_snapshot()}
    C.log(f"=== {mid} / {variant}  {os.path.basename(path)} {r['file_bytes'] / 1e9:.2f} GB  gpu={r['gpu_before']} ram={r['host_ram_before']}")
    cond = torch.load(L.cond_path(mid))
    pos, neg, seed = cond["pos"][PROMPT], cond["neg"], L.SEEDS[0]
    vae_path = fam["vae"] or prep.vae_path_for(m["ckpt"])
    C.free_all()
    model = vae = None
    try:
        t0 = time.perf_counter()
        model = C.load_diffusion(path)
        if fam.get("aura_shift"):
            model = C.aura_shift(model, fam["aura_shift"])
        r["load_s"] = time.perf_counter() - t0
        try:   # ComfyUI streams from disk (fast local storage) or from pinned RAM
            import comfy.storage
            r["fast_disk"] = comfy.storage.model_fast_disk([path])
        except Exception as e:  # pragma: no cover
            r["fast_disk"] = f"unknown: {e}"
        latent = C.empty_latent(model, *fam["latent"], L.WIDTH, L.HEIGHT)
        vae = C.load_vae(vae_path)
        counter = [0]
        am = C.with_attention(model, "pytorch", counter)

        # 2. teacher-forced accuracy check (and warm-up)
        refp = L.ref_path(mid, PROMPT)
        tf = C.TeacherForced(C.load_ref(refp)) if os.path.exists(refp) else None
        mt = am.clone()
        if tf:
            mt.set_model_unet_function_wrapper(tf)
        (lat_tf, r["warmup_s"]) = timed(lambda: C.sample(mt, pos, neg, latent, seed, m["steps"], m["cfg"],
                                                         m["sampler"], m["scheduler"]))
        if tf:
            cs = [s["cos"] for s in tf.stats]
            rs = [s["rel"] for s in tf.stats]
            r["tf"] = {"n": len(cs), "missing": tf.missing,
                       "cos_mean": sum(cs) / len(cs) if cs else None, "cos_min": min(cs) if cs else None,
                       "rel_mean": sum(rs) / len(rs) if rs else None, "rel_max": max(rs) if rs else None}
        r["residency_after_warmup"] = residency(model)
        del mt

        # 3. VAE into the cycle
        _, r["decode_first_s"] = timed(lambda: vae.decode(lat_tf))

        # 4. the timed image
        times, last = [], [None]

        mid = max(1, m["steps"] // 2)

        def step_cb(step):
            torch.cuda.synchronize()
            now = time.perf_counter()
            times.append(now - last[0])
            last[0] = now
            if len(times) == mid:
                r["gpu_mid"] = gpu_state()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        e0 = energy_mj()
        last[0] = time.perf_counter()
        start = last[0]
        out = C.sample(am, pos, neg, latent, seed, m["steps"], m["cfg"], m["sampler"], m["scheduler"], step_cb)
        torch.cuda.synchronize()
        r["sample_s"] = time.perf_counter() - start
        e1 = energy_mj()
        r["step_s"] = times
        r["residency_sampling"] = residency(model)
        r["gpu_sampling"] = C.gpu_snapshot()
        r["pcie_sampling"] = pcie_now()
        r["peak_alloc"] = torch.cuda.max_memory_allocated()
        img, r["decode_s"] = timed(lambda: vae.decode(out))
        r["image_s"] = r["sample_s"] + r["decode_s"]
        e2 = energy_mj()
        if None not in (e0, e1, e2):   # whole card, sampling + decode
            r["energy_sample_j"] = (e1 - e0) / 1000
            r["energy_image_j"] = (e2 - e0) / 1000
        r["attention_calls"] = counter[0]
        r["host_ram_after"] = meminfo()
        save_png(img, f"{d}/p{PROMPT}.png")
        st = sorted(times[1:]) or times
        res = r["residency_sampling"]
        frac = res["loaded"] / max(1, res["size"]) if "loaded" in res else None
        C.log(f"  {r['sample_s']:.2f}s sample ({st[len(st) // 2]:.3f} s/it) + {r['decode_s']:.2f}s decode, "
              f"resident {frac if frac is None else round(frac, 3)}"
              + (f", tf cos {r['tf']['cos_mean']:.5f}" if r.get("tf") else ""))
        del am
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
        r["oom"] = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
        r["traceback"] = traceback.format_exc()
        C.log(f"  ERROR {r['error']}")
    del model, vae
    C.free_all()
    C.write_json(res_p, r)


def save_png(img, path):
    from PIL import Image
    import numpy as np
    if img.ndim == 5:
        img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
    arr = (img[0].clamp(0, 1).cpu().float().numpy() * 255.0 + 0.5).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path, compress_level=4)


def read_json(p):
    import json
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def warm_cuda():
    """Pay CUDA / cuBLAS / SDPA start-up before anything is timed (the first
    GPU op in a process costs seconds to tens of seconds)."""
    import torch.nn.functional as F
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
    q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.bfloat16)
    F.scaled_dot_product_attention(q, q, q).sum().item()
    torch.cuda.synchronize()
    del x, q
    torch.cuda.empty_cache()


def child(args):
    warm_cuda()
    with torch.inference_mode():
        if args.child_te:
            te_timing(args.gpu, args.child_te)
        else:
            run_one(args.gpu, *args.child_run.split(":"))


def spawn(args, extra, what, result_path, base):
    """Run one job in a fresh process: a clean card and allocator every time,
    the way a fresh ComfyUI workflow starts, and a process the kernel kills
    for running out of RAM cannot take the rest of the pass with it."""
    import subprocess
    t0 = time.perf_counter()
    rc = subprocess.call([sys.executable, os.path.abspath(__file__), "--gpu", args.gpu, *extra])
    if rc != 0 and not os.path.exists(result_path):
        why = "killed (out of system RAM)" if rc in (-9, 137) else f"process exited with code {rc}"
        C.log(f"CRASHED {what}: {why}")
        C.write_json(result_path, {**base, "error": why, "crashed": True, "returncode": rc,
                                   "host_ram_now": meminfo()})
    C.log(f"  ({what}: {time.perf_counter() - t0:.0f}s in its own process)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True, help="card id for the output tree, e.g. rtx5080")
    ap.add_argument("--models", default="", help="model ids or family names, comma-separated")
    ap.add_argument("--variants", default="")
    ap.add_argument("--retry-crashed", action="store_true")
    ap.add_argument("--no-te", action="store_true")
    ap.add_argument("--child-te", default="", help=argparse.SUPPRESS)
    ap.add_argument("--child-run", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child_te or args.child_run:
        return child(args)
    if not C.DYNAMIC:
        C.log("note: QBENCH_DYNAMIC is not set, so this runs ComfyUI's static loader (not the default)")
    info = card_info(args.gpu)
    total = info["vram_total"]
    models = L.FIT_MODELS
    if args.models:
        want = args.models.split(",")
        models = [m for m in models if m in want or L.MODELS[m]["family"] in want]
    jobs = [(mid, v) for mid in models for v in variants_for(mid, total)
            if not args.variants or v in args.variants.split(",")]
    if args.retry_crashed:
        for mid, v in jobs:
            p = f"{run_dir(args.gpu, mid, v)}/result.json"
            if (read_json(p) or {}).get("crashed"):
                os.remove(p)
    C.log(f"{args.gpu}: {info['name']} sm_{''.join(map(str, info['capability']))}, "
          f"{total / 1e9:.1f} GB, RAM {info['host_ram']}, {len(jobs)} runs")
    fams_done = set()
    for mid, v in jobs:
        fam = L.MODELS[mid]["family"]
        if not args.no_te and fam not in fams_done:
            fams_done.add(fam)
            tp = f"{FIT}/{args.gpu}/_te/{fam}.json"
            if not os.path.exists(tp):
                spawn(args, ["--child-te", fam], f"TE {fam}", tp, {"family": fam, "gpu": args.gpu})
        rp = f"{run_dir(args.gpu, mid, v)}/result.json"
        if os.path.exists(rp):
            continue
        m = L.MODELS[mid]
        spawn(args, ["--child-run", f"{mid}:{v}"], f"{mid}/{v}", rp,
              {"model": mid, "model_name": m["name"], "family": fam, "kind": m["kind"], "variant": v,
               "gpu": args.gpu, "file": L.file_for(mid, v), "steps": m["steps"], "cfg": m["cfg"]})
    C.log("all done")


if __name__ == "__main__":
    main()
